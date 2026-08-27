"""
Composable additive-operator ROM.

Design constraint this file exists to enforce (see project discussion): Arm
2 (hand-specified operator structure, from textbook incompressible
Navier-Stokes) and Arm 3 (solver-derived operator structure, from the
extracted operator graph) MUST share the exact same model class,
per-operator hidden width, and training procedure. If they differed in
architecture as well as in structure source, a win for Arm 3 wouldn't
isolate "code-parsing helped" from "we also happened to give it a bigger
model." So there is exactly one ROM class here, and two functions that
build its `operator_specs` input from different sources. Everything else
about the two arms is identical.

Each OperatorSpec carries a `provenance` string specifically so that, when
you inspect a trained model, every block can answer "where did you come
from" -- a hand-specified textbook assumption, or a specific source line in
the solver with an extraction-confidence tag. That provenance is what
makes the eventual accuracy comparison auditable rather than a black box
result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class OperatorSpec:
    name: str
    functional_form: str   # "linear" | "quadratic" | "mlp"
    provenance: str          # human-auditable: hand-specified vs. graph node + confidence
    reads: list[str] = field(default_factory=lambda: ["*"])  # reserved for future
    # field-level masking; "*" = reads the full latent state


class QuadraticOperator(nn.Module):
    """z -> H(z, z), a per-output-dim bilinear form. Standard OpInf-style
    quadratic operator, used for the convection-type term (the physical
    convective nonlinearity is u . grad(u), quadratic in the state)."""

    def __init__(self, dim: int):
        super().__init__()
        self.bilinear = nn.Bilinear(dim, dim, dim, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.bilinear(z, z)


def _build_block(spec: OperatorSpec, latent_dim: int, hidden_dim: int) -> nn.Module:
    if spec.functional_form == "linear":
        return nn.Linear(latent_dim, latent_dim, bias=False)
    if spec.functional_form == "quadratic":
        return QuadraticOperator(latent_dim)
    if spec.functional_form == "mlp":
        return nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
    raise ValueError(f"Unknown functional_form '{spec.functional_form}' for operator '{spec.name}'")


class ComposableOperatorROM(nn.Module):
    """
    Latent dynamics dz/dt = sum_i block_i(z), one learnable block per
    OperatorSpec. Integrated with explicit Euler for training/rollout --
    swap for a proper ODE solver (torchdiffeq etc.) once this is past the
    wiring-check stage; Euler is enough to prove the architecture is
    correctly composed and trainable.
    """

    def __init__(self, latent_dim: int, operator_specs: list[OperatorSpec], hidden_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.specs = operator_specs
        self.blocks = nn.ModuleDict({
            spec.name: _build_block(spec, latent_dim, hidden_dim) for spec in operator_specs
        })

    def rhs(self, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        contributions = {name: block(z) for name, block in self.blocks.items()}
        total = sum(contributions.values())
        return total, contributions

    def rollout(self, z0: torch.Tensor, n_steps: int, dt: float) -> torch.Tensor:
        traj = [z0]
        z = z0
        for _ in range(n_steps):
            dzdt, _ = self.rhs(z)
            z = z + dt * dzdt
            traj.append(z)
        return torch.stack(traj, dim=1)  # (batch, n_steps+1, latent_dim)

    def describe(self) -> str:
        lines = [f"ComposableOperatorROM(latent_dim={self.latent_dim}, "
                 f"{len(self.specs)} operator blocks)"]
        for spec in self.specs:
            n_params = sum(p.numel() for p in self.blocks[spec.name].parameters())
            lines.append(f"  [{spec.functional_form:<9}] {spec.name:<24} "
                         f"({n_params} params)  <- {spec.provenance}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Arm 2: hand-specified operator structure (textbook incompressible NS).
# No code was read to produce this -- it's what a numerical analyst would
# write from the momentum equation alone. Kept fixed regardless of which
# solver/case is being modeled.
# --------------------------------------------------------------------------

def build_hand_specified(latent_dim: int) -> list[OperatorSpec]:
    return [
        OperatorSpec(
            "convection", "quadratic",
            provenance="hand-specified: textbook incompressible NS convective term (u . grad(u))",
        ),
        OperatorSpec(
            "diffusion", "linear",
            provenance="hand-specified: textbook incompressible NS viscous diffusion term",
        ),
        OperatorSpec(
            "pressure_gradient", "linear",
            provenance="hand-specified: textbook incompressible NS pressure-gradient coupling",
        ),
    ]


# --------------------------------------------------------------------------
# Arm 3: solver-derived operator structure, built from the extracted
# operator graph. Whatever physical_types are actually present in the
# graph become blocks here -- including anything a hand-specification
# wouldn't have anticipated (e.g. an injected/undocumented term in the
# synthetic-modification test).
# --------------------------------------------------------------------------

_PHYSICAL_TYPE_TO_FORM = {
    "convection": "quadratic",
    "diffusion": "linear",
    "pressure_gradient": "linear",
    "flux_reconstruction": "linear",
    "flux_divergence": "linear",
    "interpolation": "linear",
    "temporal_flux_correction": "linear",
    "source_implicit": "linear",
    "source_semi_implicit": "mlp",
}
_SKIP_TYPES = {"time_derivative"}  # handled by the integrator itself, not an RHS block


def build_from_operator_graph(graph, collapse_duplicates: bool = True) -> list[OperatorSpec]:
    specs: list[OperatorSpec] = []
    seen_types: set[str] = set()
    for node_id, data in graph.nodes(data=True):
        ptype = data["physical_type"]
        if ptype in _SKIP_TYPES:
            continue
        if collapse_duplicates and ptype in seen_types:
            continue  # e.g. two "diffusion"-typed nodes (velocity diffusion,
            # pressure Poisson) collapse to one latent block by default --
            # set collapse_duplicates=False to keep them as separate blocks
            # once/if the ontology distinguishes them (see ground truth
            # note on line 111).
        seen_types.add(ptype)
        form = _PHYSICAL_TYPE_TO_FORM.get(ptype, "mlp")
        conf = data.get("confidence", "unlabeled")
        specs.append(OperatorSpec(
            name=ptype,
            functional_form=form,
            provenance=(
                f"operator_graph node {node_id} ({data['qualified_name']}, "
                f"confidence={conf})"
                + ("  [UNRECOGNIZED TYPE -> mlp fallback, needs ontology entry]"
                   if ptype not in _PHYSICAL_TYPE_TO_FORM else "")
            ),
        ))
    return specs


if __name__ == "__main__":
    # Run as `python -m rom.composable_operator_rom` from the project root
    # so the `extractor` package resolves correctly.
    from extractor.operator_graph import build_operator_graph

    LATENT_DIM = 8

    graph = build_operator_graph("fixtures/icoFoam.C", "fixtures/fvSchemes")

    arm2 = ComposableOperatorROM(LATENT_DIM, build_hand_specified(LATENT_DIM))
    arm3 = ComposableOperatorROM(LATENT_DIM, build_from_operator_graph(graph))

    print("=== Arm 2: hand-specified ===")
    print(arm2.describe())
    print("\n=== Arm 3: solver-derived ===")
    print(arm3.describe())

    # Smoke test only -- synthetic data, proves the model is wired and
    # trainable end to end. This is NOT a validation result.
    print("\n=== smoke test (synthetic data, not CFD) ===")
    z0 = torch.randn(4, LATENT_DIM)
    for name, model in [("Arm 2", arm2), ("Arm 3", arm3)]:
        traj = model.rollout(z0, n_steps=10, dt=0.01)
        target = torch.randn_like(traj)
        loss = ((traj - target) ** 2).mean()
        loss.backward()
        has_grad = all(p.grad is not None for p in model.parameters())
        print(f"{name}: rollout shape={tuple(traj.shape)}, loss={loss.item():.4f}, "
              f"all params received gradients={has_grad}")
