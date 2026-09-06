"""
Experiment 1: fixed rank-15 DMD on the transient-including window.

Motivation (see status.md / vault 04): the canonical Phase 1 DMD fit selected
rank=63 by 99% cumulative energy and extrapolated to 6.03e9 on the held-out
window, with a 9.2e6 reconstruction error already inside the training window.
Question: is that blowup inherent transient instability (a linear model can't
represent nonlinear saturation, so it diverges when simulated forward -- by
mathematical necessity) or rank=63 picking up spurious noise-driven unstable
modes on top of it?

This runs fit_dmd() at an explicit low rank (15) instead of routing through
select_rank_by_energy, exactly as the canonical run does otherwise -- same
window, same split, same metrics. The three interpretable outcomes:

  * Catastrophic blowup even at rank 15 -> transient breaks linear DMD even
    when the basis is truncated hard (supports the Phase-2 thesis, but risks
    "strawman baseline" optics to a skeptical reader).
  * Moderate degradation (meaninfully worse than the rank-15 POD recon error,
    but not astronomical) -> the most useful / most credible baseline to beat.
  * Near-clean survival -> the 6.03e9 was mostly truncation pollution and
    linear DMD stays credible at low rank.

Output: results/fixed_rank_dmd.json only. Deliberately does NOT overwrite the
canonical results/ artifacts (comparison_table.csv, dmd_eigenvalues.json, ...).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch as pt

from data_loading import load_cylinder_snapshots
from dmd_baseline import fit_dmd, forecast as dmd_forecast
from pod_baseline import fit_pod
from metrics import relative_l2_error, print_summary_table

RANK = 15
T_SPLIT = 8.0


def _eigen_table(model, dt: float) -> dict:
    eig_rows = []
    n_unstable = 0
    for lam in model.dmd.eigvals:
        magnitude = abs(lam).item()
        growth_rate = float(pt.log(pt.tensor(magnitude)) / dt) if magnitude > 0 else float("-inf")
        angle = pt.angle(lam).item()
        frequency_hz = angle / (2 * pt.pi * dt) if dt > 0 else float("nan")
        if magnitude > 1.0:
            n_unstable += 1
        eig_rows.append({
            "magnitude": magnitude,
            "growth_rate": growth_rate,
            "frequency_hz": frequency_hz,
        })
    dominant = sorted(
        (r for r in eig_rows if r["magnitude"] > 1.0),
        key=lambda r: r["magnitude"], reverse=True,
    )[:5]
    return {"eigenvalues": eig_rows, "n_unstable": n_unstable, "dominant_unstable": dominant}


def main() -> None:
    snaps = load_cylinder_snapshots()
    train, test = snaps.split(t_split=T_SPLIT)
    n_train, n_test = len(train.times), len(test.times)
    print(f"Window: {len(snaps.times)} snapshots, t={snaps.times[0]}..{snaps.times[-1]}s, dt={snaps.dt}s")
    print(f"Split:  {n_train} train (t < {T_SPLIT}s) / {n_test} test (t >= {T_SPLIT}s, held out)")

    # ---- fixed low-rank DMD, bypassing energy-based rank selection ----
    model = fit_dmd(train.data_matrix, train.times, rank=RANK)
    full_forecast = dmd_forecast(model, train.data_matrix[:, 0], n_train + n_test - 1)
    recon_err = relative_l2_error(full_forecast[:, :n_train], train.data_matrix)
    extrap_err = relative_l2_error(full_forecast[:, n_train:], test.data_matrix)

    # ---- context bar: POD reconstruction error at the same rank ----
    # The rank-matched static-basis error, not the rank-63 value, is the
    # fair reference for whether rank-15 DMD "degraded moderately" vs.
    # a "phantomial" baseline (see module docstring).
    pod = fit_pod(train.data_matrix, rank=RANK)
    pod_recon_err = relative_l2_error(pod.reconstruct(RANK), train.data_matrix)

    # ---- eigenvalue survey at this rank ----
    eig = _eigen_table(model, train.dt)

    results_json = {
        "experiment": "fixed_low_rank_dmd_transient_window",
        "rank": RANK,
        "key_parameters": {
            "t_split": T_SPLIT,
            "n_train": n_train,
            "n_test": n_test,
            "dt": train.dt,
            "window": "transient-including (t_min=None)",
        },
        "dmd": {
            "reconstruction_err": recon_err,
            "extrapolation_err": extrap_err,
        },
        "pod_recon_err_at_same_rank": pod_recon_err,
        "eigenvalues": eig,
        "reference_rank63_dmd": {
            "reconstruction_err": 9228500.0,
            "extrapolation_err": 6.03e9,
        },
    }
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "fixed_rank_dmd.json"
    with open(out_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nWrote {out_path}")

    # ---- stdout comparison ----
    print(f"\nrank-{RANK} DMD:  recon err = {recon_err:.6e}   extrap err = {extrap_err:.6e}")
    print(f"rank-63 DMD:  recon err = 9.228500e+06   extrap err = 6.030000e+09")
    print(f"POD rank-{RANK} recon err (context bar): {pod_recon_err:.6e}")
    print(f"Unstable eigenvalues (|lambda| > 1) at rank {RANK}: {eig['n_unstable']} / {RANK}")
    print("Dominant unstable eigenvalues:")
    for r in eig["dominant_unstable"]:
        print(f"  |lambda|={r['magnitude']:.4f}  growth={r['growth_rate']:.3f}/s  freq={r['frequency_hz']:.3f} Hz")


if __name__ == "__main__":
    main()