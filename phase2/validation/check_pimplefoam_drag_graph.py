"""
Checks that the Step 4b drag fixture (fixtures/pimpleFoam_v2006_drag/,
UEqn.H + `fvm::Sp(cD*mag(U), U)`) parses exactly like the baseline
pimpleFoam v2006 graph (validation/check_pimplefoam_graph.py) PLUS one
extra momentum term, correctly labeled "implicit_source_nonlinear_drag"
by the operand-aware fvm::Sp rule in extractor/ontology.py, not
needs_review.

Usage:
    python -m validation.check_pimplefoam_drag_graph
"""

from __future__ import annotations

import sys

from extractor.operator_graph import build_operator_graph, equation_terms

UEQN = "fixtures/pimpleFoam_v2006_drag/UEqn.H"
PEQN = "fixtures/pimpleFoam_v2006_drag/pEqn.H"
FVSCHEMES = "fixtures/pimpleFoam_v2006_drag/fvSchemes_of_cylinder2D"
TURB_PROPS = "fixtures/pimpleFoam_v2006_drag/turbulenceProperties_of_cylinder2D"

# The 7 baseline momentum terms (identical to check_pimplefoam_graph.py's
# expected_momentum), plus the injected drag term.
EXPECTED_BASELINE = {
    ("fvm::ddt", ("U",), 1, "lhs", "time_derivative"),
    ("fvm::div", ("phi", "U"), 1, "lhs", "convection"),
    ("DDt", ("U",), 1, "lhs", "mrf_coriolis_source"),
    ("fvc::div", ("(nuEff)*dev2(T(fvc::grad(U)))",), -1, "lhs", "viscous_stress_divergence_explicit"),
    ("fvm::laplacian", ("nuEff", "U"), -1, "lhs", "diffusion"),
    ("fvOptions", ("U",), 1, "rhs", "user_source"),
    ("fvc::grad", ("p",), -1, "rhs", "pressure_gradient"),
}
EXPECTED_DRAG_TERM = ("fvm::Sp", ("cD*mag(U)", "U"), 1, "lhs", "implicit_source_nonlinear_drag")
EXPECTED_MOMENTUM = EXPECTED_BASELINE | {EXPECTED_DRAG_TERM}


def build():
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

    momentum = equation_terms(g, "UEqn")
    got_momentum = {
        (d["qualified_name"], tuple(d["arguments"]), d["sign"], d["side"], d["physical_type"])
        for _, d in momentum
    }
    if got_momentum != EXPECTED_MOMENTUM:
        failures.append(
            f"equation_terms(g, 'UEqn') mismatch:\n"
            f"  missing: {EXPECTED_MOMENTUM - got_momentum}\n"
            f"  extra:   {got_momentum - EXPECTED_MOMENTUM}"
        )

    drag_nodes = [
        (n, d) for n, d in momentum
        if d["qualified_name"] == "fvm::Sp" and tuple(d["arguments"]) == ("cD*mag(U)", "U")
    ]
    if not drag_nodes:
        failures.append("no fvm::Sp(cD*mag(U), U) node found in equation_terms(UEqn)")
    else:
        n, d = drag_nodes[0]
        if d["physical_type"] != "implicit_source_nonlinear_drag":
            failures.append(
                f"{n}: expected physical_type=implicit_source_nonlinear_drag, "
                f"got {d['physical_type']}"
            )
        if d["needs_review"]:
            failures.append(f"{n}: expected needs_review=False, got True")
        if d["sign"] != 1 or d["side"] != "lhs":
            failures.append(f"{n}: expected sign=1 side=lhs, got sign={d['sign']} side={d['side']}")

    return failures


def main() -> int:
    g = build()
    failures = check(g)
    if failures:
        print(f"FAIL ({len(failures)} issue(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"PASS: pimpleFoam_v2006_drag UEqn graph OK "
          f"({len(EXPECTED_MOMENTUM)} momentum terms including the drag term).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
