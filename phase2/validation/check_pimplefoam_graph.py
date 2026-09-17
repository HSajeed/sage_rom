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

import os
import sys
import tempfile

from extractor.operator_graph import build_operator_graph, physical_terms, equation_terms
from rom.composable_operator_rom import build_from_operator_graph

UEQN = "fixtures/pimpleFoam_v2006/UEqn.H"
PEQN = "fixtures/pimpleFoam_v2006/pEqn.H"
FVSCHEMES = "fixtures/pimpleFoam_v2006/fvSchemes_of_cylinder2D"
TURB_PROPS = "fixtures/pimpleFoam_v2006/turbulenceProperties_of_cylinder2D"
ICOFOAM_UNKNOWN = "fixtures/icoFoam_modified_unknown.C"
ICOFOAM_SCHEMES = "fixtures/fvSchemes"


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
        if d["physical_type"] != "viscous_stress_divergence_explicit":
            failures.append(
                f"{n}: expected physical_type=viscous_stress_divergence_explicit, "
                f"got {d['physical_type']}"
            )

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
        if d["physical_type"] != "pressure_gradient":
            failures.append(f"{n}: expected physical_type=pressure_gradient, got {d['physical_type']}")

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
        if d["physical_type"] != "velocity_gradient":
            failures.append(f"{n}: expected physical_type=velocity_gradient, got {d['physical_type']}")

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

    # equation_terms(g, "UEqn") is exactly the momentum equation's 7 additive
    # terms: the direct UEqn.H sum members, the two expanded viscous terms
    # (replacing the divDevReff forwarding node), and the fvc::grad(p) term
    # that only appears inside solve(UEqn == -fvc::grad(p)), never assigned
    # to tUEqn directly.
    momentum = equation_terms(g, "UEqn")
    expected_momentum = {
        ("fvm::ddt", ("U",), 1, "lhs", "time_derivative"),
        ("fvm::div", ("phi", "U"), 1, "lhs", "convection"),
        ("DDt", ("U",), 1, "lhs", "mrf_coriolis_source"),
        ("fvc::div", ("(nuEff)*dev2(T(fvc::grad(U)))",), -1, "lhs", "viscous_stress_divergence_explicit"),
        ("fvm::laplacian", ("nuEff", "U"), -1, "lhs", "diffusion"),
        ("fvOptions", ("U",), 1, "rhs", "user_source"),
        ("fvc::grad", ("p",), -1, "rhs", "pressure_gradient"),
    }
    got_momentum = {
        (d["qualified_name"], tuple(d["arguments"]), d["sign"], d["side"], d["physical_type"])
        for _, d in momentum
    }
    if got_momentum != expected_momentum:
        failures.append(
            f"equation_terms(g, 'UEqn') mismatch:\n"
            f"  missing: {expected_momentum - got_momentum}\n"
            f"  extra:   {got_momentum - expected_momentum}"
        )

    # physical_terms must never include an UNKNOWN:... node whose namespace
    # is unqualified -- those are matrix bookkeeping/control-query calls
    # (solve, relax, A, H, magSf, ...), excluded by scope
    # "bookkeeping_or_unknown_unqualified"; only genuinely unrecognized
    # QUALIFIED calls (a new/custom namespace) should surface as UNKNOWN.
    bad_unknowns = [
        (n, d) for n, d in physical_terms(g)
        if d["physical_type"].startswith("UNKNOWN:") and d["namespace"] == ""
    ]
    if bad_unknowns:
        failures.append(
            f"physical_terms has {len(bad_unknowns)} unqualified UNKNOWN node(s), "
            f"expected none: {[n for n, _ in bad_unknowns]}"
        )

    # pEqn.H physical_terms: every real operator term should survive,
    # including the two fvc::interpolate branches (one dead, per GT notes)
    # and both standalone fvc::grad(p) terms with their -=/plain-assignment
    # signs.
    pterms = physical_terms(g)
    expected_peqn = {
        ("fvc::flux", ("HbyA",), 3),
        ("fvc::interpolate", ("rAU",), 7),
        ("fvc::ddtCorr", ("U", "phi", "Uf"), 7),
        ("fvc::interpolate", ("rAU",), 11),
        ("fvc::interpolate", ("rAtU() - rAU",), 29),
        ("fvc::snGrad", ("p",), 29),
        ("fvc::grad", ("p",), 30),
        ("fvm::laplacian", ("rAtU()", "p"), 46),
        ("fvc::div", ("phiHbyA",), 46),
        ("fvc::grad", ("p",), 64),
    }
    got_peqn = {
        (d["qualified_name"], tuple(d["arguments"]), d["line_start"])
        for _, d in pterms
        if d["source_file"].endswith("pEqn.H")
    }
    missing_peqn = expected_peqn - got_peqn
    if missing_peqn:
        failures.append(f"pEqn.H physical_terms missing: {missing_peqn}")
    peqn_by_line = {(n, d["line_start"]): d for n, d in pterms if d["source_file"].endswith("pEqn.H")}
    for n, d in pterms:
        if d["source_file"].endswith("pEqn.H") and d["qualified_name"] == "fvc::grad" and d["line_start"] in (30, 64):
            if d["sign"] != -1:
                failures.append(f"{n}: expected sign=-1 for fvc::grad(p) at pEqn.H:{d['line_start']}, got {d['sign']}")

    # build_from_operator_graph(g, equation="UEqn") block names.
    momentum_specs = build_from_operator_graph(g, equation="UEqn")
    expected_block_names = {
        "convection", "mrf_coriolis_source", "viscous_stress_divergence_explicit",
        "diffusion", "user_source", "pressure_gradient",
    }
    got_block_names = {spec.name for spec in momentum_specs}
    if got_block_names != expected_block_names:
        failures.append(
            f"build_from_operator_graph(g, equation='UEqn') block names mismatch: "
            f"got {got_block_names}, expected {expected_block_names}"
        )

    return failures


def check_synthetic_unknown_term() -> list[str]:
    """
    A hand-rolled QUALIFIED custom term (customForcing::spongeSink, added
    directly to UEqn's sum in icoFoam_modified_unknown.C) must still be
    caught: kept in physical_terms and in equation_terms(UEqn) as
    UNKNOWN:..., flagged needs_review=True -- this is the detection path
    the whole ontology/labeling machinery exists to keep working even when
    a term isn't recognized.
    """
    failures = []
    g = build_operator_graph(ICOFOAM_UNKNOWN, ICOFOAM_SCHEMES, include_unqualified=True)

    def find_sponge(terms):
        return [(n, d) for n, d in terms if "spongeSink" in d["qualified_name"]]

    in_physical = find_sponge(physical_terms(g))
    if not in_physical:
        failures.append("customForcing::spongeSink missing from physical_terms")
    else:
        n, d = in_physical[0]
        if d["physical_type"] != "UNKNOWN:customForcing::spongeSink":
            failures.append(f"{n}: expected physical_type=UNKNOWN:customForcing::spongeSink, got {d['physical_type']}")
        if not d["needs_review"]:
            failures.append(f"{n}: expected needs_review=True")

    in_equation = find_sponge(equation_terms(g, "UEqn"))
    if not in_equation:
        failures.append("customForcing::spongeSink missing from equation_terms(UEqn)")

    return failures


def check_adversarial_unqualified_terms() -> list[str]:
    """
    Adversarial case (Step 4b: hand-written terms injected directly into
    solver source, not through fvm::/fvc::): two UNQUALIFIED custom terms
    -- `+ spongeSink(U)` (bare) and `+ 2.0*dampCoeff(U)` (coefficient-
    weighted) -- spliced into icoFoam.C's UEqn assembly, replacing
    `- fvm::laplacian(nu, U)` with itself plus the two new terms. Both must
    survive into equation_terms(UEqn) as UNKNOWN:..., needs_review=True,
    with the right sign; the `multiplied` factor must NOT be a reason to
    drop a term (dampCoeff has `multiplied=True`, kept anyway). Built from
    fixtures/icoFoam.C at runtime into a tempfile -- not a checked-in
    fixture -- per the coordinator's instruction.
    """
    failures = []
    src = open("fixtures/icoFoam.C").read()
    anchor = "        - fvm::laplacian(nu, U)"
    if anchor not in src:
        return [f"adversarial test anchor not found in fixtures/icoFoam.C: {anchor!r}"]
    injected = anchor + "\n      + spongeSink(U)\n      + 2.0*dampCoeff(U)"
    src2 = src.replace(anchor, injected)

    fd, tmp_path = tempfile.mkstemp(suffix=".C", dir="fixtures")
    os.close(fd)
    try:
        with open(tmp_path, "w") as f:
            f.write(src2)
        g = build_operator_graph(tmp_path, "fixtures/fvSchemes", include_unqualified=True)
        terms = dict(equation_terms(g, "UEqn"))

        def find(function):
            return [(n, d) for n, d in terms.items() if d["function"] == function]

        sponge = find("spongeSink")
        if not sponge:
            failures.append("spongeSink(U) missing from equation_terms(UEqn)")
        else:
            n, d = sponge[0]
            if d["physical_type"] != "UNKNOWN:spongeSink":
                failures.append(f"{n}: expected physical_type=UNKNOWN:spongeSink, got {d['physical_type']}")
            if not d["needs_review"]:
                failures.append(f"{n}: expected needs_review=True")
            if d["sign"] != 1:
                failures.append(f"{n}: expected sign=+1, got {d['sign']}")

        damp = find("dampCoeff")
        if not damp:
            failures.append("dampCoeff(U) missing from equation_terms(UEqn)")
        else:
            n, d = damp[0]
            if d["physical_type"] != "UNKNOWN:dampCoeff":
                failures.append(f"{n}: expected physical_type=UNKNOWN:dampCoeff, got {d['physical_type']}")
            if not d["needs_review"]:
                failures.append(f"{n}: expected needs_review=True")
            if d["sign"] != 1:
                failures.append(f"{n}: expected sign=+1, got {d['sign']}")
            if not d["multiplied"]:
                failures.append(f"{n}: expected multiplied=True (2.0*dampCoeff(U)), got {d['multiplied']}")
    finally:
        os.unlink(tmp_path)

    return failures


def summarize_momentum_terms(g) -> str:
    lines = ["Momentum-equation terms (equation_terms(g, 'UEqn')):"]
    for n, d in equation_terms(g, "UEqn"):
        lines.append(
            f"  {d['qualified_name']:<16} args={d['arguments']!s:<45} "
            f"sign={d['sign']:+d} side={str(d['side']):<5} type={d['physical_type']}"
        )
    return "\n".join(lines)


def summarize_peqn_terms(g) -> str:
    lines = ["pEqn.H physical_terms:"]
    for n, d in physical_terms(g):
        if not d["source_file"].endswith("pEqn.H"):
            continue
        lines.append(
            f"  L{d['line_start']:<3} {d['qualified_name']:<16} args={d['arguments']!s:<40} "
            f"sign={d['sign']:+d} type={d['physical_type']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    g = build()
    failures = check(g)
    failures += check_synthetic_unknown_term()
    failures += check_adversarial_unqualified_terms()
    print(summarize_momentum_terms(g))
    print()
    print(summarize_peqn_terms(g))
    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("All checks passed.")
    sys.exit(0)
