"""
Term -> regressor-family rule table for Step 4a (PATH_FORWARD.md Step 4a
specification, 2026-09-17). This is the deterministic, documented mapping
from a parsed UEqn term's `physical_type` to the OpInf regressor families
{"const", "lin", "quad"} it justifies. It does not fit anything -- it only
decides *which columns of the regressor matrix a term licenses*; opinf.py
builds and fits those columns.

Rule table (physical_type -> families):
  time_derivative                         -> LHS, no regressor family (dz/dt or z_{k+1} IS the target)
  convection            (div(phi,U))       -> quad   (phi = flux(U) is linear in U, so div(phi,U) ~ quadratic in U)
  diffusion             (laplacian(nu,U))  -> lin    (nu constant -> linear in U; requires a laminar,
                                                       constant-coefficient case, checked below)
  viscous_stress_divergence_explicit       -> lin    (same constant-nuEff argument, laminar case only)
  pressure_gradient     (grad(p))          -> const + lin + quad (p solves a Poisson equation with a
                                                       source quadratic in U, plus boundary/const terms)
  mrf_coriolis_source                      -> dropped if MRF inactive, else raise (not modeled -- Step 4b territory)
  user_source           (fvOptions)        -> dropped if fvOptions inactive, else raise (not modeled -- Step 4b territory)
  const                                    -> always included (inlet BC; centred POD coordinates)

Any `needs_review` term, or any term whose physical_type isn't in this
table, makes the builder refuse to run -- Arm 3 must never silently guess.

Arm 2 (textbook) is the same rule table applied to the fixed physical_type
set {convection, diffusion, pressure_gradient}, independent of any parsed
graph -- it is what a textbook incompressible-NS discretisation would give.
Arm 3 (`arm3_library`) applies the identical rule table to the terms the
parser actually finds in `equation_terms(g, "UEqn")`, filtered by
`case_activity.py`. On this (unmodified, laminar, no-MRF, no-fvOptions)
dataset the two arms are required to produce the identical family set --
this is the null control Step 4a is built to run, asserted in
run_step4a.py, not here.

Out-of-span "extra" families (Step 4b: e.g. a `quadratic_drag` regressor
Phi^T(|U~|U~) for an injected `fvm::Sp(cD*mag(U), U)` term) are looked up
through EXTRA_FAMILY_REGISTRY, a name -> callable(Z, basis_ctx) -> columns
hook. Nothing is registered here yet -- Step 4b adds the callable, not new
plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .case_activity import CaseActivity, read_case_activity

FAMILY_ORDER = ("const", "lin", "quad")

# physical_type -> frozenset of regressor families it licenses. Terms not
# listed here are either LHS (time_derivative, handled separately) or
# conditional on case activity (_CONDITIONAL_RULES below) or unmapped
# (-> raise).
_STATIC_RULES: dict[str, frozenset[str]] = {
    "convection": frozenset({"quad"}),
    "diffusion": frozenset({"lin"}),
    "viscous_stress_divergence_explicit": frozenset({"lin"}),
    "pressure_gradient": frozenset({"const", "lin", "quad"}),
}

# physical_type -> case_activity flag name. If the flag is False, the term
# is dropped (its families contribute nothing). If True, the builder
# raises: Step 4a has no rule for an ACTIVE MRF/fvOptions term (that's
# Step 4b's `user_source` extension), so silently guessing a family would
# be worse than refusing.
_CONDITIONAL_RULES: dict[str, str] = {
    "mrf_coriolis_source": "mrf_active",
    "user_source": "fvoptions_active",
}

# physical_type values requiring the laminar/constant-coefficient
# assumption behind the diffusion and viscous_stress_divergence_explicit
# rules above.
_CONSTANT_COEFFICIENT_TYPES = frozenset({"diffusion", "viscous_stress_divergence_explicit"})

_LHS_TYPES = frozenset({"time_derivative"})

EXTRA_FAMILY_REGISTRY: dict[str, Callable] = {}


def register_extra_family(name: str, fn: Callable) -> None:
    """Registers an out-of-span regressor family builder:
    fn(Z, basis_ctx) -> Tensor of shape (k, n_samples). `basis_ctx` is
    whatever the caller building regressors needs to reconstruct the full
    field from Z (POD modes/mean) -- e.g. Step 4b's quadratic_drag needs
    the reconstructed velocity to form |U~|U~ pointwise before projecting
    back. Nothing is registered by default; this hook exists so Step 4b
    only adds a rule table entry + one callable, not new plumbing."""
    EXTRA_FAMILY_REGISTRY[name] = fn


@dataclass
class Library:
    families: tuple[str, ...]        # sorted subset of FAMILY_ORDER
    extras: tuple[str, ...]          # names looked up in EXTRA_FAMILY_REGISTRY
    provenance: list[dict] = field(default_factory=list)


def _sorted_families(families: set[str]) -> tuple[str, ...]:
    return tuple(f for f in FAMILY_ORDER if f in families)


def arm2_library() -> Library:
    """Textbook list: convection, diffusion, pressure_gradient -> {const, lin, quad}."""
    families: set[str] = set()
    provenance = []
    for physical_type in ("convection", "diffusion", "pressure_gradient"):
        term_families = _STATIC_RULES[physical_type]
        families |= term_families
        provenance.append({
            "qualified_name": f"textbook:{physical_type}",
            "arguments": [],
            "physical_type": physical_type,
            "families": sorted(term_families),
            "status": "included",
        })
    return Library(families=_sorted_families(families), extras=(), provenance=provenance)


def arm3_library(graph, equation: str = "UEqn", case_dir: str | None = None,
                  case_activity: CaseActivity | None = None) -> Library:
    """Builds the Arm 3 library from `equation_terms(graph, equation)`,
    filtered by the case's activity flags. Raises ValueError on any
    `needs_review` term, any term whose physical_type isn't in the rule
    table, or any active MRF/fvOptions term (Step 4a has no rule for those
    yet -- see module docstring)."""
    from extractor.operator_graph import equation_terms

    if case_activity is None:
        if case_dir is None:
            raise ValueError("arm3_library requires case_dir or case_activity")
        case_activity = read_case_activity(case_dir)

    families: set[str] = set()
    provenance: list[dict] = []

    for node_id, data in equation_terms(graph, equation):
        physical_type = data["physical_type"]
        row = {
            "node_id": node_id,
            "qualified_name": data["qualified_name"],
            "arguments": list(data["arguments"]),
            "physical_type": physical_type,
        }

        if data.get("needs_review"):
            raise ValueError(
                f"arm3_library: term {data['qualified_name']}({data['arguments']}) "
                f"[{node_id}] is needs_review=True -- refusing to build a library "
                f"with an unreviewed term."
            )

        if physical_type in _LHS_TYPES:
            row.update(families=[], status="lhs")
            provenance.append(row)
            continue

        if physical_type in _CONDITIONAL_RULES:
            flag_name = _CONDITIONAL_RULES[physical_type]
            active = getattr(case_activity, flag_name)
            if active:
                raise ValueError(
                    f"arm3_library: term {data['qualified_name']}({data['arguments']}) "
                    f"[{node_id}] is physical_type={physical_type!r} and case_activity."
                    f"{flag_name}=True -- Step 4a has no regressor rule for an ACTIVE "
                    f"{physical_type} term (that is Step 4b's user_source extension); "
                    f"refusing to silently drop or guess a family for it."
                )
            row.update(families=[], status=f"dropped ({flag_name}=False)")
            provenance.append(row)
            continue

        if physical_type not in _STATIC_RULES:
            raise ValueError(
                f"arm3_library: term {data['qualified_name']}({data['arguments']}) "
                f"[{node_id}] has unmapped physical_type={physical_type!r} -- no rule "
                f"in term_library._STATIC_RULES/_CONDITIONAL_RULES."
            )

        if physical_type in _CONSTANT_COEFFICIENT_TYPES and case_activity.simulation_type != "laminar":
            raise ValueError(
                f"arm3_library: term {data['qualified_name']}({data['arguments']}) "
                f"[{node_id}] is physical_type={physical_type!r}, which this rule table "
                f"only licenses as 'lin' under the laminar, constant-coefficient "
                f"assumption; case_activity.simulation_type={case_activity.simulation_type!r}."
            )

        term_families = _STATIC_RULES[physical_type]
        families |= term_families
        row.update(families=sorted(term_families), status="included")
        provenance.append(row)

    # const is always included (inlet BC; centred POD coordinates).
    families.add("const")

    return Library(families=_sorted_families(families), extras=(), provenance=provenance)


if __name__ == "__main__":
    lib2 = arm2_library()
    print("arm2:", lib2.families)
