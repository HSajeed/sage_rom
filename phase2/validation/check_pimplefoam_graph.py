"""
Checks that the pimpleFoam v2006 momentum equation is correctly assembled
by the operator graph once unqualified-call capture and dispatch expansion
are turned on: the viscous term (turbulence->divDevReff(U)) must expand to
its two linearViscousStress.C terms with the right signs, and the terms
already directly visible in UEqn.H must carry the right sign/side.

Usage:
    python -m validation.check_pimplefoam_graph
"""

from __future__ import annotations

import sys

from extractor.operator_graph import build_operator_graph, physical_terms

UEQN = "fixtures/pimpleFoam_v2006/UEqn.H"
PEQN = "fixtures/pimpleFoam_v2006/pEqn.H"
FVSCHEMES = "fixtures/pimpleFoam_v2006/fvSchemes_of_cylinder2D"
TURB_PROPS = "fixtures/pimpleFoam_v2006/turbulenceProperties_of_cylinder2D"


def build() -> "nx.DiGraph":
    return build_operator_graph(
        UEQN,
        fvschemes_path=FVSCHEMES,
        extra_sources=[PEQN],
        include_unqualified=True,
        expand_dispatch=True,
        turbulence_properties_path=TURB_PROPS,
    )


def check(g) -> list[str]:
    failures = []

    def find(pred):
        return [(n, d) for n, d in g.nodes(data=True) if pred(n, d)]

    laplacian_nuEff_U = find(
        lambda n, d: d["qualified_name"] == "fvm::laplacian"
        and d["arguments"] == ["nuEff", "U"]
    )
    if not laplacian_nuEff_U:
        failures.append("no fvm::laplacian(nuEff, U) node found")
    else:
        n, d = laplacian_nuEff_U[0]
        if d["sign"] != -1:
            failures.append(f"{n}: expected sign=-1, got {d['sign']}")
        if not d["expanded_from"] or "divDevReff" not in d["expanded_from"]:
            failures.append(f"{n}: expected expanded_from divDevReff, got {d['expanded_from']}")

    div_dev2 = find(
        lambda n, d: d["qualified_name"] == "fvc::div"
        and d["arguments"] == ["(nuEff)*dev2(T(fvc::grad(U)))"]
    )
    if not div_dev2:
        failures.append("no fvc::div((nuEff)*dev2(T(fvc::grad(U)))) node found")
    else:
        n, d = div_dev2[0]
        if d["sign"] != -1:
            failures.append(f"{n}: expected sign=-1, got {d['sign']}")

    ddt_u = find(lambda n, d: d["qualified_name"] == "fvm::ddt" and d["arguments"] == ["U"])
    if not ddt_u:
        failures.append("no fvm::ddt(U) node found")
    else:
        n, d = ddt_u[0]
        if d["sign"] != 1:
            failures.append(f"{n}: expected sign=+1, got {d['sign']}")

    grad_p_rhs = find(
        lambda n, d: d["qualified_name"] == "fvc::grad"
        and d["arguments"] == ["p"]
        and d["source_file"].endswith("UEqn.H")
    )
    if not grad_p_rhs:
        failures.append("no fvc::grad(p) node found in UEqn.H")
    else:
        n, d = grad_p_rhs[0]
        if d["side"] != "rhs" or d["sign"] != -1:
            failures.append(f"{n}: expected side=rhs sign=-1, got side={d['side']} sign={d['sign']}")

    terms = physical_terms(g)
    term_ids = {n for n, _ in terms}

    divdevreff = find(lambda n, d: d["qualified_name"] == "divDevReff" and d["receiver"] == "turbulence")
    if not divdevreff:
        failures.append("no divDevReff(turbulence) node found")
    else:
        n, d = divdevreff[0]
        if not d["expanded"]:
            failures.append(f"{n}: expected expanded=True, got {d['expanded']}")
        if n in term_ids:
            failures.append(f"{n}: expected NOT in physical_terms (it was replaced by its expansion)")

    nested_grad_u = find(
        lambda n, d: d["qualified_name"] == "fvc::grad"
        and d["arguments"] == ["U"]
        and d["source_file"].endswith("linearViscousStress.C")
        and d["line_start"] == 102
    )
    if not nested_grad_u:
        failures.append("no expanded fvc::grad(U) node found (linearViscousStress.C:102)")
    else:
        n, d = nested_grad_u[0]
        if not d["nested_in"]:
            failures.append(f"{n}: expected nested_in to be set, got {d['nested_in']}")
        if n in term_ids:
            failures.append(f"{n}: expected NOT in physical_terms (it is nested inside fvc::div's argument)")

    # Robustness check for the byte-range nesting fix: pEqn.H has two
    # textually identical standalone fvc::grad(p) terms (lines 30 and 64),
    # neither of which is anyone's argument -- raw-text substring matching
    # could wrongly mark one "nested" in the other; byte ranges can't.
    standalone_grad_p = find(
        lambda n, d: d["qualified_name"] == "fvc::grad"
        and d["arguments"] == ["p"]
        and d["source_file"].endswith("pEqn.H")
        and d["line_start"] in (30, 64)
    )
    if len(standalone_grad_p) != 2:
        failures.append(f"expected 2 standalone fvc::grad(p) nodes in pEqn.H (lines 30, 64), found {len(standalone_grad_p)}")
    for n, d in standalone_grad_p:
        if d["nested_in"] is not None:
            failures.append(f"{n}: expected nested_in=None (standalone term), got {d['nested_in']}")

    return failures


def summarize_momentum_terms(g) -> str:
    lines = ["Momentum-equation terms (physical_terms only -- UEqn.H + expanded viscous term):"]
    for n, d in physical_terms(g):
        if d["assigned_to"] != "tUEqn" and d["expanded_from"] is None:
            continue
        if d["source_file"].endswith("pEqn.H"):
            continue
        lines.append(
            f"  {d['qualified_name']:<16} args={d['arguments']!s:<45} "
            f"sign={d['sign']:+d} side={str(d['side']):<5} type={d['physical_type']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    g = build()
    failures = check(g)
    print(summarize_momentum_terms(g))
    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("All checks passed.")
    sys.exit(0)
