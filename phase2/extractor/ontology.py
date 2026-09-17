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
            "Implicit source/sink term, linear in the solved field."
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
            "PROPOSED -- needs human review. Surface-normal gradient "
            "evaluated on faces (e.g. the SIMPLEC pressure-flux term in "
            "pEqn.H); not yet confirmed against upstream source by a human "
            "reviewer."
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
