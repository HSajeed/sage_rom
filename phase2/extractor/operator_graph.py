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
from .ontology import lookup, is_known_namespace
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
        argument of another batch-member call -- e.g. `fvc::grad(U)` inside
        `fvc::div(... dev2(T(fvc::grad(U))))`'s argument. Position-based, not
        text-based: a call is nested in another iff they're in the same
        source_file and the child's [start_byte, end_byte) lies strictly
        inside the parent's own argument_list [args_start_byte,
        args_end_byte). Raw-text substring matching would wrongly conflate
        two textually identical calls on different lines (e.g. two separate
        `fvc::grad(p)` terms) -- byte ranges from tree-sitter can't do that.
        When several candidate parents contain the child (e.g. both `T(...)`
        and the outer `fvc::div(...)`), the smallest (innermost) containing
        argument_list wins.
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
                g.nodes[node_id]["nested_in"] = candidates[0][1]

    def _add_call_node(call: ExtractedCall, expanded_from: str | None = None, provenance: str | None = None) -> str:
        node_id = _node_id(call)
        meaning = lookup(call.namespace, call.function)
        block = _BLOCK_FOR_FUNCTION.get(call.function)
        scheme = (
            scheme_for(scheme_entries, block, _scheme_key(call))
            if (block and scheme_entries) else None
        )

        # Unrecognized calls get a physical_type that includes their
        # qualified name, not a flat "UNKNOWN" -- a flat string means
        # build_from_operator_graph's collapse_duplicates (on by default)
        # would silently merge calls from DIFFERENT unrecognized namespaces
        # into ONE ROM block, as if a customForcing::spongeSink term and an
        # unrelated customPhysics::lorentzForce term were the same physical
        # operator just because neither was in the ontology yet. Found via
        # external review of this exact code path, not caught by the
        # existing tests (which only ever exercised one unknown namespace
        # at a time). Each unrecognized call now gets its own type until a
        # human or the LLM labeler actually determines two of them ARE the
        # same thing.
        physical_type = meaning.physical_type if meaning else f"UNKNOWN:{call.qualified_name}"

        g.add_node(
            node_id,
            namespace=call.namespace,
            function=call.function,
            qualified_name=call.qualified_name,
            arguments=call.arguments,
            physical_type=physical_type,
            discretization_role=meaning.discretization_role if meaning else "UNKNOWN",
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
    terms, not the call site that dispatched to them) and nodes that are
    themselves a sub-expression argument of another captured call
    (`nested_in` set -- e.g. the `fvc::grad(U)` inside
    `fvc::div(...dev2(T(fvc::grad(U))))` is part of that div's argument, not
    a separate additive term of the equation). For graphs built without
    expand_dispatch/nesting detection producing any matches (e.g. the
    icoFoam graph), both attributes are False/None on every node, so this
    returns every node -- i.e. build_from_operator_graph's behaviour is
    unchanged when there's nothing to filter.
    """
    return [
        (node_id, data) for node_id, data in g.nodes(data=True)
        if not data.get("expanded", False) and data.get("nested_in") is None
    ]


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
