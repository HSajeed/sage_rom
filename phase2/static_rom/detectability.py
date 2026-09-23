"""
Pre-run detectability check for a Step 4b injected term (run BEFORE
spending an OpenFOAM run on it).

An injected term can only separate Arm 3 from Arm 2 through the part of
its regressor that lies outside Arm 2's span {1, z, z⊗z} on the data
manifold. This projects the candidate regressor G(z) (evaluated on the
unmodified dataset's POD coefficients) onto that span and compares the
out-of-span remainder, scaled by the injection amplitude and the
snapshot interval, with Arm 2's 4a validation residual:

    signal_r   = dt * amplitude * ||G_perp|| / ||z_{k+1}||
    detectable = signal_r > B_val * val_residual_arm2(r)   (B_val from
                                                            DECISION_RULE_step5.md)

This is a first-order estimate: it ignores how the injected term changes
the flow itself, and it assumes the drag-free data manifold.

Run as a module from phase2/:
    python -m static_rom.detectability --family quadratic_drag --amplitude 0.3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch as pt

_PHASE2_DIR = Path(__file__).resolve().parent.parent
_PHASE1_DIR = _PHASE2_DIR.parent / "phase1"
if str(_PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASE1_DIR))

from data_loading import load_cylinder_snapshots                          # noqa: E402

from static_rom.opinf import (                                            # noqa: E402
    build_regressors, build_target, make_basis_ctx, pod_basis, project,
)
from static_rom.term_library import EXTRA_FAMILY_REGISTRY, Library      # noqa: E402

RANKS = (8, 11, 15, 23)
T_SPLIT = 8.0
B_VAL = 3.0


def _arm2_val_residuals(grid_csv: Path) -> dict[tuple[str, int], float]:
    out = {}
    with open(grid_csv) as f:
        for row in csv.DictReader(f):
            if row["arm"] == "arm2" and row["target"] == "discrete":
                out[(row["window"], int(row["rank"]))] = float(row["val_residual"])
    return out


def detectability(family: str, amplitude: float, grid_csv: Path) -> list[dict]:
    snaps = load_cylinder_snapshots()
    train, _test = snaps.split(T_SPLIT)
    windows = {"full": train, "post_transient": train.split(4.0)[1]}
    arm2 = Library(families=("const", "lin", "quad"), extras=())
    val_res = _arm2_val_residuals(grid_csv)
    rows = []
    for window_name, w in windows.items():
        for rank in RANKS:
            basis = pod_basis(w.data_matrix, rank)
            Z = project(basis, w.data_matrix)
            G = EXTRA_FAMILY_REGISTRY[family](Z, make_basis_ctx(basis, snaps.n_cells_selected))
            D, _groups = build_regressors(Z, arm2)
            G_perp = G - pt.linalg.lstsq(D.T, G.T).solution.T @ D
            G_fluct = G - G.mean(dim=1, keepdim=True)
            _Z_in, Y, _idx, _trunc = build_target(Z, "discrete")
            signal = (w.dt * amplitude * G_perp[:, :-1].norm() / Y.norm()).item()
            residual = val_res[(window_name, rank)]
            rows.append({
                "window": window_name, "rank": rank,
                "out_of_span_fraction": (G_perp.norm() / G_fluct.norm()).item(),
                "signal": signal, "arm2_val_residual": residual,
                "signal_over_threshold": signal / (B_VAL * residual),
                "detectable": signal > B_VAL * residual,
            })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4b pre-run detectability of an injected term.")
    parser.add_argument("--family", default="quadratic_drag", help="extra-family name in EXTRA_FAMILY_REGISTRY")
    parser.add_argument("--amplitude", type=float, required=True, help="injection coefficient (e.g. cD)")
    parser.add_argument("--grid", default=str(_PHASE2_DIR / "results_step4a" / "grid.csv"))
    parser.add_argument("--out", default=None, help="optional JSON output path")
    args = parser.parse_args()

    rows = detectability(args.family, args.amplitude, Path(args.grid))
    for r in rows:
        print(f"{r['window']:>14} r={r['rank']:>2}  out-of-span={r['out_of_span_fraction']:.2e}  "
              f"signal={r['signal']:.2e}  arm2 val_res={r['arm2_val_residual']:.2e}  "
              f"signal/(B_val*res)={r['signal_over_threshold']:.2e}  detectable={r['detectable']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"family": args.family, "amplitude": args.amplitude, "b_val": B_VAL, "rows": rows}, f, indent=2)
