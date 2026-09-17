"""
Explicit expansion table for virtual/runtime-selected calls the AST alone
cannot resolve -- the extractor sees `turbulence->divDevReff(U)`, but which
concrete function that dispatches to depends on the case's
constant/turbulenceProperties (simulationType, and the laminar/RAS/LES model
selected within it), not on anything visible at UEqn.H's call site.

Each entry here is hand-curated and cites exactly where it came from (source
file + line, plus any intermediate forwarding call), so a wrong expansion is
auditable and fixable in one place. This is deliberately NOT a general
virtual-dispatch resolver -- see project scope discussion in ast_parser.py's
module docstring for why the extractor targets a small, stylized DSL rather
than a full C++ call graph.

Currently one entry:
  incompressible::momentumTransportModel::divDevReff, simulationType=laminar,
  laminar model=Stokes, OpenFOAM.com v2006
    -> turbulence->divDevReff(U)                         [UEqn.H:9]
       forwards to IncompressibleTurbulenceModel::divDevReff(U), which
       forwards to the virtual divDevRhoReff(U)            [IncompressibleTurbulenceModel.C:117]
       Stokes has no divDevRhoReff override (see Stokes.C/.H in this
       fixture), so it inherits linearViscousStress::divDevRhoReff(U),
       whose body is the actual expansion                  [linearViscousStress.C:102-103]
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .ast_parser import ExtractedCall, extract_calls

_LAMINAR_MODEL_RE = re.compile(r"\blaminar\s*\{([^}]*)\}", re.DOTALL)
_SIMULATION_TYPE_RE = re.compile(r"\bsimulationType\s+([^;\s]+)\s*;")


@dataclass
class TurbulenceSelection:
    simulation_type: str          # e.g. "laminar", "RAS", "LES"
    laminar_model: str | None     # e.g. "Stokes"; None if not laminar


def parse_turbulence_properties(path: str) -> TurbulenceSelection:
    """
    Minimal parser for constant/turbulenceProperties, in the same spirit as
    fvschemes_parser.py: only extracts the two keys dispatch.py needs
    (simulationType, and the laminar{} model subdict if present), not a
    general OpenFOAM dictionary parser.
    """
    with open(path) as f:
        text = f.read()

    sim_match = _SIMULATION_TYPE_RE.search(text)
    simulation_type = sim_match.group(1) if sim_match else "laminar"

    laminar_model = None
    if simulation_type == "laminar":
        lam_match = _LAMINAR_MODEL_RE.search(text)
        if lam_match:
            model_match = re.search(r"\bmodel\s+([^;\s]+)\s*;", lam_match.group(1))
            if model_match:
                laminar_model = model_match.group(1)
        if laminar_model is None:
            # v2006's turbulenceProperties for this dataset case (see
            # fixtures/pimpleFoam_v2006/turbulenceProperties_of_cylinder2D)
            # has simulationType laminar with no laminar{} subdict at all.
            # laminarModels::Stokes is OpenFOAM's documented default laminar
            # model in that situation (confirmed against
            # src/TurbulenceModels/turbulenceModels/laminar/Stokes/Stokes.H
            # in this fixture: "Description: Turbulence model for Stokes
            # flow", the no-op laminar model). Not independently verified
            # against the laminar-model *selection* source (the
            # runTimeSelectable constructor table / laminarModel.C), so
            # this default is an assumption, flagged here for review.
            laminar_model = "Stokes"

    return TurbulenceSelection(simulation_type=simulation_type, laminar_model=laminar_model)


def _normalize_arg(arg: str) -> str:
    """
    Strip the identity factors OpenFOAM's incompressible turbulence-model
    template carries around for the compressible/multiphase-shared code path
    but which are no-ops (geometricOneField) here: `this->alpha_*this->rho_*`
    and `this->nuEff()` -> `nuEff` (drop the `this->`, since the expanded
    terms are being attributed to the case's momentum equation, not left as
    a still-implicit member reference).
    """
    arg = arg.replace("this->alpha_*this->rho_*", "")
    arg = arg.replace("this->nuEff()", "nuEff")
    arg = arg.replace("this->", "")
    return arg


@dataclass
class DispatchExpansion:
    calls: list[ExtractedCall]
    provenance: str


# Explicit table: (receiver_type, method, simulation_type, laminar_model,
# version) -> function that builds the expansion. Only one entry exists
# today; add more (RAS models, other methods) as separate entries rather
# than generalizing prematurely.
_PROVENANCE = (
    "turbulence->divDevReff(U) [UEqn.H:9] -> "
    "IncompressibleTurbulenceModel::divDevReff -> divDevRhoReff (virtual) "
    "[IncompressibleTurbulenceModel.C:117] -> Stokes inherits "
    "linearViscousStress::divDevRhoReff(U) "
    "[linearViscousStress.C:102-103], OpenFOAM.com v2006 "
    "(see fixtures/pimpleFoam_v2006/PROVENANCE.txt)"
)


def _expand_incompressible_stokes_divDevReff(
    outer: ExtractedCall, fixtures_dir: str
) -> DispatchExpansion:
    source_path = os.path.join(fixtures_dir, "linearViscousStress.C")
    all_calls = extract_calls(source_path, known_namespaces=None)
    # The U-only overload's body is lines 102-103 (see PROVENANCE.txt); the
    # (rho, U) overload at 118-119 is a different, unused overload.
    body_calls = [c for c in all_calls if c.line_start in (102, 103)]

    expanded: list[ExtractedCall] = []
    for c in body_calls:
        expanded.append(ExtractedCall(
            namespace=c.namespace,
            function=c.function,
            arguments=[_normalize_arg(a) for a in c.arguments],
            raw_text=c.raw_text,
            source_file=c.source_file,
            line_start=c.line_start,
            line_end=c.line_end,
            assigned_to=outer.assigned_to,
            receiver=c.receiver,
            access=c.access,
            sign=c.sign * outer.sign,
            side=outer.side,
            multiplied=c.multiplied,
            start_byte=c.start_byte,
            end_byte=c.end_byte,
            args_start_byte=c.args_start_byte,
            args_end_byte=c.args_end_byte,
        ))
    return DispatchExpansion(calls=expanded, provenance=_PROVENANCE)


# key: (receiver_type, method, simulation_type, laminar_model, version)
_DISPATCH_TABLE = {
    (
        "incompressible::momentumTransportModel/turbulence",
        "divDevReff",
        "laminar",
        "Stokes",
        "OpenFOAM.com v2006",
    ): _expand_incompressible_stokes_divDevReff,
}


def expand(
    outer: ExtractedCall,
    receiver_type: str,
    selection: TurbulenceSelection,
    fixtures_dir: str,
    version: str = "OpenFOAM.com v2006",
) -> DispatchExpansion | None:
    """
    Look up and expand `outer` (e.g. the ExtractedCall for
    `turbulence->divDevReff(U)`) against the dispatch table, given the
    case's turbulence-model selection. Returns None if there's no matching
    entry (the call is left as-is by the caller).
    """
    key = (receiver_type, outer.function, selection.simulation_type, selection.laminar_model, version)
    builder = _DISPATCH_TABLE.get(key)
    if builder is None:
        return None
    return builder(outer, fixtures_dir)
