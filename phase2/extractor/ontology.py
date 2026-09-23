"""
Fixed physical-operator ontology for the SAGE-CFD extractor.

Design intent (see project discussion): the AST extractor should not need an
LLM to identify *standard* fvm::/fvc::/fvi:: operators, because OpenFOAM's
finite-volume DSL already names the physics in the function name itself.
This module is the deterministic, checkable core of that claim. An LLM (or a
human) only gets involved for calls that DON'T match anything here -- custom
source terms, hand-rolled loops, or namespaces the ontology hasn't seen yet
(this file's TODO list at the bottom exists precisely to catch that: the
real icoFoam.C fixture used a namespace, `fvi::`, that classic OpenFOAM
teaching material does not mention, which is exactly the kind of drift this
whole pipeline is meant to catch instead of silently missing).

Each entry maps a DSL call to:
  - physical_type : the term's role in the discretized PDE
  - discretization_role : implicit ("fvm") vs explicit ("fvc"/"fvi") treatment
  - notes : short justification, for provenance / human review
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorMeaning:
    physical_type: str
    discretization_role: str
    notes: str


# namespace -> function name -> meaning
# discretization_role:
#   "implicit"   -> fvm:: : contributes to the matrix (Ax=b) being solved
#   "explicit"   -> fvc:: : evaluated from the current field, not part of the matrix
#   "interpolated" -> fvi:: : face-interpolated / reconstruction-style operator
ONTOLOGY: dict[str, dict[str, OperatorMeaning]] = {
    "fvm": {
        "ddt": OperatorMeaning(
            "time_derivative", "implicit",
            "Implicit time derivative term (unsteady/transient operator)."
        ),
        "div": OperatorMeaning(
            "convection", "implicit",
            "Implicit divergence -- the convective transport term, "
            "typically div(phi, U) for momentum advection by flux phi."
        ),
        "laplacian": OperatorMeaning(
            "diffusion", "implicit",
            "Implicit Laplacian -- diffusive term (viscous stress or "
            "scalar diffusion, depending on the field it acts on)."
        ),
        "Sp": OperatorMeaning(
            "source_implicit", "implicit",
            "Implicit source/sink term, linear in the solved field. "
            "Operand-aware callers should use resolve_label() instead, "
            "which distinguishes 'implicit_source_linear' (constant "
            "coefficient) from 'implicit_source_nonlinear_drag' "
            "(coefficient contains mag() of the solved field, e.g. "
            "fvm::Sp(cD*mag(U), U) -- Step 4b's injected drag term)."
        ),
        "SuSp": OperatorMeaning(
            "source_semi_implicit", "implicit",
            "Semi-implicit source term (blends explicit and implicit "
            "treatment depending on sign)."
        ),
    },
    "fvc": {
        "grad": OperatorMeaning(
            "pressure_gradient_explicit", "explicit",
            "Explicit gradient, evaluated from the current field -- "
            "commonly used for correction terms, not the primary "
            "pressure-gradient coupling (compare fvi::grad in this fixture)."
        ),
        "div": OperatorMeaning(
            "convection_explicit", "explicit",
            "Explicit divergence, e.g. of a face flux field."
        ),
        "flux": OperatorMeaning(
            "flux_reconstruction", "explicit",
            "Reconstructs a face flux field from a volume field "
            "(e.g. phiHbyA = fvc::flux(HbyA))."
        ),
        "interpolate": OperatorMeaning(
            "interpolation", "explicit",
            "Cell-to-face interpolation of a volume field."
        ),
        "ddtCorr": OperatorMeaning(
            "temporal_flux_correction", "explicit",
            "Correction term reconciling the face flux with the time "
            "derivative (PISO/PIMPLE flux consistency)."
        ),
        "reconstruct": OperatorMeaning(
            "field_reconstruction", "explicit",
            "Reconstructs a volume field (typically a vector) from a "
            "face-flux-like quantity."
        ),
        "snGrad": OperatorMeaning(
            "surface_normal_gradient", "explicit",
            "Confirmed: Programmer's Guide v2512 Sec 3.4.6 eq 3.28 and "
            "Table 3.2 cover the surface-normal gradient. Operand-aware "
            "callers should use resolve_label() instead, which returns "
            "'pressure_gradient_face_normal' for snGrad(p)."
        ),
        "makeRelative": OperatorMeaning(
            "mesh_motion_flux_adjustment", "explicit",
            "PROPOSED -- needs human review. Subtracts the mesh motion "
            "flux from a face flux field (ALE/moving-mesh consistency); "
            "not yet confirmed against upstream source by a human reviewer."
        ),
        "makeAbsolute": OperatorMeaning(
            "mesh_motion_flux_adjustment", "explicit",
            "PROPOSED -- needs human review. Adds the mesh motion flux "
            "back to a face flux field (inverse of fvc::makeRelative); "
            "not yet confirmed against upstream source by a human reviewer."
        ),
        "correctUf": OperatorMeaning(
            "mesh_motion_flux_adjustment", "explicit",
            "PROPOSED -- needs human review. Corrects the face velocity "
            "field Uf for a moving mesh; not yet confirmed against "
            "upstream source by a human reviewer."
        ),
    },
    "fvi": {
        # CONFIRMED via upstream OpenFOAM-dev source
        # (src/finiteVolume/finiteVolume/fvi/: fviGrad.H, fviDiv.H/.C, plus
        # fviDdt/Laplacian/Reconstruct/Sup), declared `InNamespace Foam::fvi`.
        # The header docstrings state these operators return a
        # `volInternalField` -- i.e. an internal-cell-only evaluation, with
        # no boundary-face treatment, unlike fvc:: which returns a full
        # volField (internal + boundary). This is a genuine, distinct
        # discretization-role difference from fvc::, not just a renamed
        # duplicate -- confirmed by comparing against OpenFOAM v11's
        # icoFoam.C (still fvc::grad/fvc::div at this call site) against
        # OpenFOAM-dev's icoFoam.C (fvi::grad/fvi::div at the same site,
        # `U.internalFieldRef() = HbyA() - rAU()*fvi::grad(p)` vs v11's
        # `U = HbyA - rAU*fvc::grad(p)`) -- the internalFieldRef() on the
        # LHS is the tell: dev is deliberately working in internal-field-only
        # space, matching what fvi:: is documented to return.
        "grad": OperatorMeaning(
            "pressure_gradient", "explicit_internal",
            "Confirmed: gradient evaluated on internal cells only "
            "(volInternalField, no boundary-face treatment), used where "
            "the surrounding code is already working in internal-field "
            "space (see U.internalFieldRef() at the call site)."
        ),
        "div": OperatorMeaning(
            "flux_divergence", "explicit_internal",
            "Confirmed: divergence evaluated on internal cells only "
            "(volInternalField). Labeled flux_divergence rather than "
            "generic 'convection' since this appears in the pressure "
            "equation acting on a face-flux field (phiHbyA), not on a "
            "momentum-convection term -- still an open human-review "
            "question (see ground truth notes) whether physical_type "
            "should be split further by which equation the divergence "
            "appears in, not just which namespace computed it."
        ),
    },
}


def lookup(namespace: str, function: str) -> OperatorMeaning | None:
    return ONTOLOGY.get(namespace, {}).get(function)


def is_known_namespace(namespace: str) -> bool:
    return namespace in ONTOLOGY


# ---------------------------------------------------------------------------
# Operand-aware labeling (owner policy 2026-09-17, see
# validation/ground_truth_pimpleFoam_v2006.yaml top-of-file rules and
# validation/labeling_notes_pimpleFoam_v2006.md "Labeling policy" /
# "Ontology changes implied"). `resolve_label` supersedes the flat
# (namespace, function) -> meaning table above for callers that can supply
# arguments/receiver -- it decides physical_type = operator + operand
# (what the term IS in context), not just the DSL call name.
#
# Known limits (documented, not fixed): operand detection below is purely
# syntactic string matching on the normalized argument text -- it has no
# field-type information. A flux field that doesn't follow the `phi*`
# naming convention won't be recognized as a flux; a non-flux field that
# happens to be named `phi*` will be misdetected as one. Same caveat for
# the `nuEff`/`dev`/`dev2` substring check used to spot the viscous-stress
# argument of fvc::div.
# ---------------------------------------------------------------------------


def _normalize_operand(text: str) -> str:
    """
    Simple syntactic normalization of an argument's source text for operand
    matching: strip surrounding whitespace, drop internal spaces, strip
    `this->` member-access prefixes, and drop a trailing empty-call `()`
    (e.g. `this->nuEff()` -> `nuEff`, `this->U_` -> `U_`).
    """
    a = text.strip().replace(" ", "").replace("this->", "")
    if a.endswith("()"):
        a = a[:-2]
    return a


def _is_pressure_field(operand: str) -> bool:
    return operand in ("p", "p_rgh")


def _is_velocity_field(operand: str) -> bool:
    return operand in ("U", "U_")


def _looks_like_face_flux(operand: str) -> bool:
    """Known limit: naming-convention heuristic only, see module note above."""
    return operand.startswith("phi")


def _looks_like_viscous_stress_arg(operand: str) -> bool:
    """Known limit: substring heuristic only, see module note above."""
    return "nuEff" in operand or "dev2(" in operand or "dev(" in operand


def _resolve_grad(function: str, arguments: list[str]) -> OperatorMeaning | None:
    operand = _normalize_operand(arguments[0]) if arguments else ""
    if function == "grad":
        if _is_pressure_field(operand):
            return OperatorMeaning(
                "pressure_gradient", "explicit",
                "Operand-aware: fvc::grad(p) is the pressure-gradient term "
                "(matches the reviewed icoFoam fvi::grad(p) label)."
            )
        if _is_velocity_field(operand):
            return OperatorMeaning(
                "velocity_gradient", "explicit",
                "Operand-aware: fvc::grad(U) -- gradient of the velocity "
                "field, not the pressure-gradient coupling term."
            )
        return OperatorMeaning(
            "gradient", "explicit",
            "Operand-aware: generic fvc::grad of a field that is neither "
            "the pressure nor the velocity."
        )
    if function == "snGrad":
        if _is_pressure_field(operand):
            return OperatorMeaning(
                "pressure_gradient_face_normal", "explicit",
                "Operand-aware: fvc::snGrad(p), the face-normal pressure "
                "gradient (e.g. the SIMPLEC flux-consistency term)."
            )
        return OperatorMeaning(
            "surface_normal_gradient", "explicit",
            "Operand-aware: fvc::snGrad of a non-pressure field."
        )
    return None


def _resolve_fvm_sp(arguments: list[str]) -> OperatorMeaning:
    """
    Operand-aware fvm::Sp(coeff, field): a plain fvm::Sp(lambda, U) with a
    coefficient that does not reference the solved field is a linear
    implicit source/sink (coeff*U, coeff constant in U). fvm::Sp(cD*mag(U),
    U) -- Step 4b's injected drag term -- has a coefficient containing
    mag(<field>) of the very field being solved, so the term coeff*U =
    cD*mag(U)*U is quadratic (drag) in U even though the fvMatrix
    contribution is assembled implicitly; it is given its own physical_type
    so term_library.py can route it to the out-of-span quadratic_drag
    regressor family instead of the linear one.
    """
    coeff = _normalize_operand(arguments[0]) if arguments else ""
    field = _normalize_operand(arguments[1]) if len(arguments) > 1 else ""
    if field and re.search(rf"mag\(\s*{re.escape(field)}\s*\)", coeff):
        return OperatorMeaning(
            "implicit_source_nonlinear_drag", "implicit",
            "Operand-aware: fvm::Sp(coeff, U) whose coefficient contains "
            "mag(U) of the solved field U -- an implicit quadratic-drag "
            "sink (coeff*U with |U|-dependent coeff), not a constant-"
            "coefficient linear source."
        )
    return OperatorMeaning(
        "implicit_source_linear", "implicit",
        "Operand-aware: fvm::Sp(coeff, U) with a coefficient that does not "
        "depend on the solved field's magnitude -- linear in U."
    )


def _resolve_fvc_div(arguments: list[str]) -> OperatorMeaning:
    if len(arguments) == 2:
        return OperatorMeaning(
            "convection", "explicit",
            "Operand-aware: two-argument fvc::div(flux, field) is the "
            "explicit convective transport term."
        )
    operand = _normalize_operand(arguments[0]) if arguments else ""
    if _looks_like_viscous_stress_arg(operand):
        return OperatorMeaning(
            "viscous_stress_divergence_explicit", "explicit",
            "Operand-aware: single-argument fvc::div of a viscous-stress "
            "tensor expression (contains nuEff/dev/dev2)."
        )
    return OperatorMeaning(
        "flux_divergence", "explicit",
        "Operand-aware: single-argument fvc::div of a face-flux-typed "
        "field (surfaceIntegrate), distinct from convection per the "
        "guide's Divergence vs Convection sections."
    )


# receiver-keyed table for unqualified (namespace == "") operator calls,
# see labeling_notes_pimpleFoam_v2006.md "Ontology changes implied" #7-#8.
_RECEIVER_TABLE: dict[tuple[str | None, str], OperatorMeaning] = {
    ("MRF", "DDt"): OperatorMeaning(
        "mrf_coriolis_source", "explicit",
        "MRFZoneList::DDt(U) -- Coriolis acceleration source, explicit "
        "volField added to the matrix."
    ),
    (None, "fvOptions"): OperatorMeaning(
        "user_source", "implicit_or_explicit",
        "fv::optionList::operator()(U) builds an fvMatrix from each "
        "option's addSup -- not necessarily explicit."
    ),
    ("fvOptions", "fvOptions"): OperatorMeaning(
        "user_source", "implicit_or_explicit",
        "fv::optionList::operator()(U) builds an fvMatrix from each "
        "option's addSup -- not necessarily explicit."
    ),
    ("turbulence", "divDevReff"): OperatorMeaning(
        "viscous_stress_divergence", "implicit+explicit",
        "Forwarding node (Rule 3): the whole deviatoric viscous term, "
        "expanded by dispatch.py into its implicit/explicit parts."
    ),
    (None, "divDevRhoReff"): OperatorMeaning(
        "viscous_stress_divergence", "implicit+explicit",
        "Forwarding node (Rule 3): same physical term as divDevReff, "
        "must not be counted as a second viscous term."
    ),
    (None, "nuEff"): OperatorMeaning(
        "effective_viscosity", "coefficient",
        "Coefficient (Gamma of the Laplacian / the viscous-stress "
        "argument), not a discretized operator."
    ),
    ("this", "nuEff"): OperatorMeaning(
        "effective_viscosity", "coefficient",
        "Coefficient (Gamma of the Laplacian / the viscous-stress "
        "argument), not a discretized operator."
    ),
    (None, "dev2"): OperatorMeaning(
        "deviatoric_part_2", "algebraic",
        "dev2(A) = A - (2/3)tr(A)I, pointwise tensor algebra."
    ),
    (None, "T"): OperatorMeaning(
        "tensor_transpose", "algebraic",
        "Pointwise tensor transpose, (A^T)_ij = A_ji."
    ),
}

# Unqualified/qualified calls that are bookkeeping (matrix/flux
# manipulation, mesh-motion frame changes on a static mesh, etc.), never an
# operator label. Returning None here (rather than a "bookkeeping"
# physical_type) keeps resolve_label's contract simple -- "None" already
# means "this call is not an operator label the ontology assigns", exactly
# as it does for any other unrecognized call; detect_modification.py's
# "UNRECOGNIZED -> route to LLM/human review" framing still applies (these
# calls just happen to be legitimately non-physical, not merely unmapped).
_BOOKKEEPING_FUNCTIONS = {
    ("fvc", "makeRelative"), ("fvc", "makeAbsolute"), ("fvc", "correctUf"),
}


def is_bookkeeping_call(namespace: str, function: str) -> bool:
    """
    True for calls resolve_label deliberately labels as non-operator
    bookkeeping (fvc::makeRelative/makeAbsolute/correctUf) -- lets a caller
    (operator_graph.py) distinguish "known bookkeeping" from "genuinely
    unrecognized" even though resolve_label returns None for both.
    """
    return (namespace, function) in _BOOKKEEPING_FUNCTIONS


def resolve_label(
    namespace: str,
    function: str,
    arguments: list[str],
    receiver: str | None = None,
    access: str | None = None,
) -> OperatorMeaning | None:
    """
    Operand-aware physical_type/discretization_role resolution: physical_type
    = operator + operand, per the owner labeling policy (see module docstring
    above this function). Falls back to the flat `lookup()` table for
    namespace/function combinations that have no operand-aware rule (fvm::,
    fvi::, and the remaining fvc:: entries) so all previously-labeled call
    sites keep their existing label unchanged.
    """
    if (namespace, function) in _BOOKKEEPING_FUNCTIONS:
        return None

    if namespace == "fvc" and function in ("grad", "snGrad"):
        meaning = _resolve_grad(function, arguments)
        if meaning is not None:
            return meaning

    if namespace == "fvc" and function == "div":
        return _resolve_fvc_div(arguments)

    if namespace == "fvm" and function == "Sp":
        return _resolve_fvm_sp(arguments)

    if not namespace:
        meaning = _RECEIVER_TABLE.get((receiver, function))
        if meaning is not None:
            return meaning

    return lookup(namespace, function)
