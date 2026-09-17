"""
Builds the CFD operator graph by fusing two sources:

  1. AST-extracted fvm::/fvc::/fvi:: calls (ast_parser.py)   -- WHICH
     operators exist, what fields they act on, and how the resulting terms
     are assembled into fvVectorMatrix/fvScalarMatrix objects (from
     `assigned_to`). This comes from the solver's C++ source and is
     case-independent.

  2. fvSchemes discretization entries (fvschemes_parser.py)    -- HOW each
     operator is actually discretized for a given case (e.g. Gauss linear
     vs Gauss upwind for div(phi,U)). This is case configuration, not code,
     and changes per case without recompiling the solver.

A node's identity (which physical operator it is) comes only from source
(1); a node's `scheme` attribute comes from source (2) when a case is
supplied, and is left as None for a solver-only (no case) graph.

Sequencing edges encode the PISO predictor-corrector structure. This is
NOT inferred from the AST -- it's a known, documented algorithm skeleton
(see project discussion: don't re-discover what's already textbook). The
AST/LLM work fills in the leaves (which operators, which schemes), not the
top-level control structure.
"""

from __future__ import annotations

import networkx as nx

from .ast_parser import ExtractedCall, extract_calls
from .fvschemes_parser import SchemeEntry, parse_fvschemes, scheme_for
from .ontology import resolve_label, is_known_namespace, is_bookkeeping_call
from . import dispatch as dispatch_mod


# Map a namespace's fvSchemes block name, so a fvm::laplacian(...) call
# knows to look itself up under 'laplacianSchemes', etc.
_BLOCK_FOR_FUNCTION = {
    "ddt": "ddtSchemes",
    "div": "divSchemes",
    "laplacian": "laplacianSchemes",
    "grad": "gradSchemes",
    "interpolate": "interpolationSchemes",
}


def _scheme_key(call: ExtractedCall) -> str:
    """Reconstruct the fvSchemes lookup key, e.g. 'div(phi,U)'."""
    return f"{call.function}({','.join(a.replace(' ', '') for a in call.arguments)})"


def build_operator_graph(
    source_path: str,
    fvschemes_path: str | None = None,
    piso_skeleton: bool = True,
    extra_sources: list[str] | None = None,
    include_unqualified: bool = False,
    expand_dispatch: bool = False,
    turbulence_properties_path: str | None = None,
) -> nx.DiGraph:
    """
    `extra_sources`: additional files (e.g. pEqn.H alongside UEqn.H) whose
    calls are folded into the same graph -- default None keeps single-file
    behaviour unchanged.
    `include_unqualified`: forwarded to extract_calls (default off).
    `expand_dispatch`/`turbulence_properties_path`: if both given, calls
    matching the dispatch.py table (currently `turbulence->divDevReff`) get
    additional nodes for their runtime-resolved expansion, tagged
    `expanded_from`/`provenance`; default off leaves the graph unchanged.
    """
    import os

    all_sources = [source_path] + list(extra_sources or [])
    multi_file = len(all_sources) > 1

    # None = capture every qualified (namespace::function) call, not just
    # fvm/fvc/fvi. Narrowing to known namespaces belongs at the ontology
    # lookup stage (which correctly flags unknowns), not at extraction --
    # filtering here would silently drop exactly the custom-namespace
    # modifications this pipeline exists to catch (see
    # validation/detect_modification.py's unknown-namespace test).
    calls: list[ExtractedCall] = []
    for src in all_sources:
        calls.extend(extract_calls(src, known_namespaces=None, include_unqualified=include_unqualified))
    scheme_entries: list[SchemeEntry] = (
        parse_fvschemes(fvschemes_path) if fvschemes_path else []
    )

    turbulence_selection = (
        dispatch_mod.parse_turbulence_properties(turbulence_properties_path)
        if (expand_dispatch and turbulence_properties_path) else None
    )

    g = nx.DiGraph()

    def _node_id(call: ExtractedCall) -> str:
        if multi_file:
            return f"{os.path.basename(call.source_file)}::{call.function}_{call.line_start}"
        return f"{call.function}_{call.line_start}"

    def _assign_nesting(batch: list[tuple[str, ExtractedCall]]) -> None:
        """
        Mark a node `nested_in: <parent node id>` when it is a sub-expression
        argument of another batch-member call that is itself an operator
        (resolve_label(...) is not None -- see `is_operator` on the node) --
        e.g. `fvc::grad(U)` inside `fvc::div(... dev2(T(fvc::grad(U))))`'s
        argument. A call nested inside a non-operator wrapper (`solve`,
        `MRF.zeroFilter`, `constrainHbyA`, `max`, `tmp`/`.ref()`, ...) is
        transparent: walk outward past it to the innermost *operator*
        ancestor instead of demoting the call to that wrapper's argument
        (see GT notes: nesting inside solve/zeroFilter/constrainHbyA does
        not demote a term). If no containing call resolves as an operator,
        `nested_in` stays None and the call remains a standalone term.

        Position-based, not text-based: a call is nested in another iff
        they're in the same source_file and the child's [start_byte,
        end_byte) lies strictly inside the parent's own argument_list
        [args_start_byte, args_end_byte). Raw-text substring matching would
        wrongly conflate two textually identical calls on different lines
        (e.g. two separate `fvc::grad(p)` terms) -- byte ranges from
        tree-sitter can't do that. When several candidate parents contain
        the child (e.g. both `T(...)` and the outer `fvc::div(...)`), the
        smallest (innermost) containing argument_list that IS an operator
        wins.
        """
        for node_id, call in batch:
            candidates: list[tuple[int, str]] = []
            for other_id, other_call in batch:
                if other_id == node_id:
                    continue
                if other_call.source_file != call.source_file:
                    continue
                a_start, a_end = other_call.args_start_byte, other_call.args_end_byte
                if a_start is None or a_end is None:
                    continue
                if a_start < call.start_byte and call.end_byte < a_end:
                    candidates.append((a_end - a_start, other_id))
            if candidates:
                candidates.sort(key=lambda t: t[0])
                for _, cand_id in candidates:
                    if g.nodes[cand_id].get("is_operator"):
                        g.nodes[node_id]["nested_in"] = cand_id
                        break

    def _add_call_node(call: ExtractedCall, expanded_from: str | None = None, provenance: str | None = None) -> str:
        node_id = _node_id(call)
        meaning = resolve_label(
            call.namespace, call.function, call.arguments,
            receiver=call.receiver, access=call.access,
        )
        block = _BLOCK_FOR_FUNCTION.get(call.function)
        scheme = (
            scheme_for(scheme_entries, block, _scheme_key(call))
            if (block and scheme_entries) else None
        )

        is_bk = is_bookkeeping_call(call.namespace, call.function)
        if is_bk:
            # Explicitly-recognized non-operator bookkeeping (fvc::
            # makeRelative/makeAbsolute/correctUf on a static mesh): a real
            # label the ontology assigns, distinct from "genuinely
            # unrecognized" -- excluded from physical_terms below, but not
            # flagged for review.
            physical_type = "bookkeeping"
            discretization_role = "bookkeeping"
            scope = "bookkeeping"
            needs_review = False
        elif meaning is not None:
            physical_type = meaning.physical_type
            discretization_role = meaning.discretization_role
            scope = "operator"
            needs_review = False
        else:
            # Unrecognized calls get a physical_type that includes their
            # qualified name, not a flat "UNKNOWN" -- a flat string means
            # build_from_operator_graph's collapse_duplicates (on by
            # default) would silently merge calls from DIFFERENT
            # unrecognized namespaces into ONE ROM block, as if a
            # customForcing::spongeSink term and an unrelated
            # customPhysics::lorentzForce term were the same physical
            # operator just because neither was in the ontology yet. Found
            # via external review of this exact code path, not caught by
            # the existing tests (which only ever exercised one unknown
            # namespace at a time). Each unrecognized call now gets its own
            # type until a human or the LLM labeler actually determines two
            # of them ARE the same thing.
            physical_type = f"UNKNOWN:{call.qualified_name}"
            discretization_role = "UNKNOWN"
            needs_review = True
            # Qualified unrecognized calls (a new/custom namespace) stay a
            # physical_terms member -- that's the detection path this
            # pipeline exists to keep working. Unqualified unrecognized
            # calls (plain member/free-function calls picked up only via
            # include_unqualified: solve, relax, A, H, magSf, ...) are
            # mostly matrix bookkeeping/control queries, not equation terms
            # -- physical_terms() excludes them by default; equation_terms()
            # below re-admits the ones that are genuine direct additive
            # terms of a specific equation's solve(...) expression.
            scope = "unknown_qualified" if call.namespace else "bookkeeping_or_unknown_unqualified"

        g.add_node(
            node_id,
            namespace=call.namespace,
            function=call.function,
            qualified_name=call.qualified_name,
            arguments=call.arguments,
            physical_type=physical_type,
            discretization_role=discretization_role,
            scope=scope,
            needs_review=needs_review,
            is_operator=(meaning is not None),
            confidence="low" if (meaning and "LOW CONFIDENCE" in meaning.notes) else (
                "unlabeled" if meaning is None else "high"
            ),
            scheme=scheme,
            assigned_to=call.assigned_to,
            source_file=call.source_file,
            line_start=call.line_start,
            line_end=call.line_end,
            receiver=call.receiver,
            access=call.access,
            sign=call.sign,
            side=call.side,
            multiplied=call.multiplied,
            start_byte=call.start_byte,
            end_byte=call.end_byte,
            args_start_byte=call.args_start_byte,
            args_end_byte=call.args_end_byte,
            expanded_from=expanded_from,
            provenance=provenance,
            expanded=False,
            nested_in=None,
        )
        return node_id

    main_batch: list[tuple[str, ExtractedCall]] = []
    for call in calls:
        node_id = _add_call_node(call)
        main_batch.append((node_id, call))

        if (
            expand_dispatch and turbulence_selection is not None
            and call.receiver == "turbulence" and call.function == "divDevReff"
        ):
            expansion = dispatch_mod.expand(
                call,
                receiver_type="incompressible::momentumTransportModel/turbulence",
                selection=turbulence_selection,
                fixtures_dir=os.path.dirname(source_path),
            )
            if expansion is not None:
                # The dispatched-to call (e.g. turbulence->divDevReff(U)) is
                # replaced by its expansion -- it stays in the graph for
                # provenance/audit, but is no longer itself a term of the
                # equation, so physical_terms() must exclude it.
                g.nodes[node_id]["expanded"] = True
                expansion_batch: list[tuple[str, ExtractedCall]] = []
                for expanded_call in expansion.calls:
                    exp_node_id = _add_call_node(
                        expanded_call, expanded_from=node_id, provenance=expansion.provenance
                    )
                    expansion_batch.append((exp_node_id, expanded_call))
                _assign_nesting(expansion_batch)

    _assign_nesting(main_batch)

    # Composition edges: terms that flow into the same LHS (e.g. ddt(U),
    # div(phi,U), laplacian(nu,U) all -> UEqn) sum into one equation.
    by_target: dict[str, list[str]] = {}
    for node_id, data in g.nodes(data=True):
        target = data.get("assigned_to")
        if target:
            by_target.setdefault(target, []).append(node_id)
    for target, members in by_target.items():
        for a, b in zip(members, members[1:]):
            g.add_edge(a, b, kind="composition", target=target)

    # Data-flow edges: if node X's target variable (e.g. "UEqn") appears as
    # an argument to a later call, or if a later call's target variable
    # name appears in an earlier argument list (e.g. phiHbyA used inside
    # fvi::div(phiHbyA)), link them. Shallow, syntactic, not full alias
    # analysis -- sufficient to seed the graph for review.
    targets_seen: dict[str, str] = {}  # var name -> node_id that produced it
    for node_id, data in g.nodes(data=True):
        target = data.get("assigned_to")
        if target:
            targets_seen[target] = node_id
    for node_id, data in g.nodes(data=True):
        for arg in data.get("arguments", []):
            arg_clean = arg.strip()
            if arg_clean in targets_seen and targets_seen[arg_clean] != node_id:
                g.add_edge(targets_seen[arg_clean], node_id, kind="data_flow", via=arg_clean)

    # Equation aliases: `fvVectorMatrix& UEqn = tUEqn.ref();` is a syntactic
    # alias -- UEqn and tUEqn name the same equation. Recognized pattern:
    # an unqualified `.ref()`/`->ref()` call whose receiver is the
    # equation's tmp<> variable and whose assigned_to is a reference
    # declarator (`& UEqn`, `&UEqn`). Maps both directions so
    # equation_terms() can be called with either spelling.
    aliases: dict[str, str] = {}
    for _, data in g.nodes(data=True):
        if data.get("function") == "ref" and data.get("namespace") == "" and data.get("receiver"):
            declared = (data.get("assigned_to") or "").lstrip("&").strip()
            if declared:
                aliases[declared] = data["receiver"]
                aliases[data["receiver"]] = data["receiver"]
    g.graph["equation_aliases"] = aliases

    if piso_skeleton:
        g.graph["algorithm_skeleton"] = [
            "momentum_predictor",   # UEqn assembled + optionally solved
            "piso_loop",             # while (piso.correct())
            "pressure_corrector",    # pEqn assembled + solved
            "flux_correction",       # phi = phiHbyA - pEqn.flux()
            "velocity_correction",   # U reconstructed from HbyA, rAU, grad(p)
        ]
        g.graph["skeleton_source"] = (
            "Hand-specified from the documented PISO algorithm, NOT "
            "inferred from the AST. See module docstring."
        )

    return g


def physical_terms(g: nx.DiGraph) -> list[tuple[str, dict]]:
    """
    The graph's leaf equation terms: excludes nodes replaced by a dispatch
    expansion (`expanded=True` -- the expansion's own nodes are the real
    terms, not the call site that dispatched to them), nodes that are
    themselves a sub-expression argument of another captured OPERATOR call
    (`nested_in` set -- e.g. the `fvc::grad(U)` inside
    `fvc::div(...dev2(T(fvc::grad(U))))` is part of that div's argument, not
    a separate additive term of the equation), and nodes that are
    unqualified (namespace "") calls resolve_label doesn't recognize as an
    operator (`scope == "bookkeeping_or_unknown_unqualified"` --
    fvMatrix/tmp<> accessors, control queries, solve/relax/constrain
    wrappers picked up only via include_unqualified; a genuine unqualified
    custom term would still need to appear via equation_terms() below,
    which re-admits exactly the direct additive members of one equation)
    and nodes resolve_label recognizes as explicit non-operator bookkeeping
    (`scope == "bookkeeping"` -- fvc::makeRelative/makeAbsolute/correctUf).
    Qualified calls the ontology doesn't recognize (a new/custom namespace,
    e.g. customForcing::spongeSink) are KEPT here, tagged
    `physical_type="UNKNOWN:..."` and `needs_review=True` -- that's the
    detection/review path this pipeline exists to keep working. For graphs
    built without expand_dispatch/nesting detection producing any matches
    (e.g. the icoFoam graph, whose 10 entries are all qualified fvm/fvc/fvi
    calls with `expanded`/`nested_in` False/None and `scope="operator"` on
    every node), this returns every node -- i.e. build_from_operator_graph's
    behaviour is unchanged when there's nothing to filter.
    """
    return [
        (node_id, data) for node_id, data in g.nodes(data=True)
        if not data.get("expanded", False)
        and data.get("nested_in") is None
        and data.get("scope") not in ("bookkeeping", "bookkeeping_or_unknown_unqualified")
    ]


# Unqualified calls that are never additive equation terms even though they
# take >=1 argument -- matrix/field bookkeeping and arithmetic helpers, not
# physics. Documented list, not exhaustive: everything else with >=1
# argument that isn't nested inside an operator and isn't itself a
# reference to the equation object is treated as a possible custom term
# (see equation_terms docstring -- Step 4b will inject hand-written terms
# into solver source, so admission is deliberately permissive here).
_NON_TERM_UNQUALIFIED_FUNCTIONS = {
    "solve", "max", "min", "mag", "sqr", "ref", "tmp", "relax", "correct",
    "constrain", "A", "H", "H1", "flux", "magSf", "Sf",
}


def _normalize_arg_text(arg: str) -> str:
    a = arg.strip().replace(" ", "").replace("this->", "")
    if a.endswith("()"):
        a = a[:-2]
    return a


def _is_admissible_unqualified_term(data: dict, names: set[str]) -> bool:
    """
    True if an unqualified-unresolved call (`scope ==
    "bookkeeping_or_unknown_unqualified"`) looks like a genuine additive
    equation term rather than matrix bookkeeping/an accessor: it has at
    least one argument (rAU()/magSf()-style zero-arg accessors are
    excluded by this alone), its function isn't in the documented
    non-term list above, it isn't nested inside another operator call's
    argument list, and none of its own arguments is a reference to the
    equation object itself (e.g. `fvOptions.constrain(UEqn)` takes the
    equation as its argument -- that's a mutation of an existing term, not
    a new one).
    """
    if data.get("expanded", False) or data.get("nested_in") is not None:
        return False
    if data.get("namespace") != "" or data.get("scope") != "bookkeeping_or_unknown_unqualified":
        return False
    args = data.get("arguments") or []
    if not args:
        return False
    if data.get("function") in _NON_TERM_UNQUALIFIED_FUNCTIONS:
        return False
    if any(_normalize_arg_text(a) in names for a in args):
        return False
    return True


def equation_terms(g: nx.DiGraph, target: str) -> list[tuple[str, dict]]:
    """
    The additive terms of ONE discretized equation named `target` (e.g.
    "UEqn"/"tUEqn", "pEqn") -- a superset of the intersection of
    physical_terms(g) with that equation, scoped to a single fvMatrix
    rather than every equation in the graph.

    `target` may be given as either the declared tmp<> variable
    (`tUEqn`) or a `.ref()` alias of it (`UEqn`, from
    `fvVectorMatrix& UEqn = tUEqn.ref();`); both resolve to the same
    equation via `g.graph["equation_aliases"]`.

    A node belongs to the equation if either:
      (a) its `assigned_to` is the equation's declared variable (the
          `tUEqn = fvm::ddt(U) + ...` assembly) -- this already covers
          expanded dispatch terms, which inherit the outer call's
          assigned_to; or
      (b) it lies inside a `solve(<target-or-alias> == ...)` call's
          argument expression (e.g. `solve(UEqn == -fvc::grad(p))`), by
          byte-range containment.
    In both cases, a node that's already a physical_terms() member is
    included unconditionally -- a term multiplied by a coefficient is
    still a real equation term, matching how `sign`/`side` already treat
    multiplication as pass-through. A node that's an unqualified-unresolved
    call (excluded from physical_terms globally, since most of those are
    matrix bookkeeping) is re-admitted in EITHER location if it looks like
    a genuine additive term (see `_is_admissible_unqualified_term`) --
    Gate 1 can't recognize hand-written custom terms by name, so it must
    not matter whether they're written inside the equation's own
    assembly (`tUEqn = ... + spongeSink(U) + 2.0*dampCoeff(U)`) or inside
    solve(...); both are tagged `physical_type="UNKNOWN:..."`,
    `needs_review=True`, and keep their computed `sign`/`multiplied` (a
    `2.0*dampCoeff(U)`-style coefficient factor is still an additive term,
    just flagged `multiplied=True` for review, not dropped).
    """
    aliases = g.graph.get("equation_aliases", {})
    canonical = aliases.get(target, target)
    names = {target, canonical}

    pt = dict(physical_terms(g))
    result: dict[str, dict] = {}

    for node_id, data in g.nodes(data=True):
        if data.get("assigned_to") not in names:
            continue
        if node_id in pt:
            result[node_id] = data
        elif _is_admissible_unqualified_term(data, names):
            result[node_id] = data

    solve_calls = [
        (nid, d) for nid, d in g.nodes(data=True)
        if d.get("namespace") == "" and d.get("function") == "solve" and d.get("arguments")
    ]
    for _, sdata in solve_calls:
        arg0 = sdata["arguments"][0].replace(" ", "")
        if not any(arg0.startswith(name + "==") for name in names):
            continue
        a_start, a_end = sdata.get("args_start_byte"), sdata.get("args_end_byte")
        if a_start is None or a_end is None:
            continue
        for onid, odata in g.nodes(data=True):
            if odata.get("source_file") != sdata.get("source_file"):
                continue
            osb, oeb = odata.get("start_byte"), odata.get("end_byte")
            if osb is None or oeb is None:
                continue
            if not (a_start < osb and oeb < a_end):
                continue
            if odata.get("expanded", False) or odata.get("nested_in") is not None:
                continue
            if odata.get("scope") == "bookkeeping_or_unknown_unqualified":
                if _is_admissible_unqualified_term(odata, names):
                    result[onid] = odata
                continue
            if odata.get("scope") == "bookkeeping":
                continue
            result[onid] = odata

    return list(result.items())


def summarize(g: nx.DiGraph) -> str:
    lines = [f"Operator graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges"]
    if "algorithm_skeleton" in g.graph:
        lines.append("Algorithm skeleton: " + " -> ".join(g.graph["algorithm_skeleton"]))
    lines.append("")
    for node_id, data in g.nodes(data=True):
        conf_flag = f" [{data['confidence'].upper()}]" if data["confidence"] != "high" else ""
        lines.append(
            f"  {node_id:<16} {data['qualified_name']:<16} "
            f"type={data['physical_type']:<28} scheme={str(data['scheme']):<20}"
            f"{conf_flag}"
        )
    lines.append("")
    for u, v, data in g.edges(data=True):
        lines.append(f"  {u} --[{data['kind']}]--> {v}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "fixtures/icoFoam.C"
    schemes = sys.argv[2] if len(sys.argv) > 2 else "fixtures/fvSchemes"
    graph = build_operator_graph(src, schemes)
    print(summarize(graph))
