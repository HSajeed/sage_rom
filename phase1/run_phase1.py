"""
Phase 1 orchestration. Loads the real cylinder2D dataset (see
data_loading.py for setup), splits into a training window and a held-out
extrapolation window, fits POD / DMD / the neural ROM baseline, and prints
the comparison table across the four metrics.

RUN THIS YOURSELF once flowTorch and the dataset are set up -- this
scaffold verifies each component against synthetic data (see
test_synthetic.py and each module's own `if __name__` block) but does not
download or touch the real dataset. Before trusting any number this
prints, sanity-check it the way this scaffold's own development did: does
DMD recover eigenvalues on (or very near) the unit circle at plausible
shedding frequencies? Does POD's reconstruction error actually decrease
with rank? A baseline that "runs" is not the same as a baseline that's
right -- ask it something you can independently sanity-check before
trusting the comparison table.
"""

from __future__ import annotations

import csv
import json
import math
import pickle
from datetime import datetime
from pathlib import Path

import torch as pt

from data_loading import load_cylinder_snapshots, CylinderSnapshots
from pod_baseline import fit_pod, select_rank_by_energy, reconstruction_error_vs_rank
from dmd_baseline import fit_dmd, forecast as dmd_forecast
from neural_rom_baseline import fit_neural_rom, forecast as neural_forecast
from metrics import relative_l2_error, time_call, data_efficiency_curve, print_summary_table


def run(snaps: CylinderSnapshots, t_split: float, latent_dim: int = 8) -> dict:
    train, test = snaps.split(t_split)
    n_train, n_test = len(train.times), len(test.times)
    print(f"Train: {n_train} snapshots (t < {t_split}s)   "
          f"Test: {n_test} snapshots (t >= {t_split}s, held out)")

    rank = select_rank_by_energy(train.data_matrix, energy_threshold=0.99)
    print(f"Rank selected at 99% cumulative energy: {rank}  "
          f"(DMD and POD-reconstruction-at-full-rank both use this)")

    results: dict = {}

    # ---- POD: reconstruction quality vs. rank only, no forecast (see
    # pod_baseline.py's module docstring for why) ----
    (pod, pod_fit_time) = time_call(fit_pod, train.data_matrix, rank)
    recon_err = relative_l2_error(pod.reconstruct(rank), train.data_matrix)
    results["POD"] = {
        "reconstruction_err": recon_err,
        "extrapolation_err": None,   # honest n/a, not a bug -- static basis, no dynamics
        "fit_time_s": pod_fit_time,
    }

    # ---- DMD ----
    def dmd_forecast_with_n(n: int) -> pt.Tensor:
        sub_data, sub_times = train.data_matrix[:, :n], train.times[:n]
        # Reselect rank from THIS subset, not the full training set's rank
        # capped by n-1. Reusing a rank chosen from 210 snapshots (e.g. 90)
        # at n=20 means fitting a 19-dimensional linear operator off 19
        # transitions -- severely overfit and numerically unstable (an
        # earlier version of this function did exactly that and produced a
        # data-efficiency curve with an extrapolation error of 461 at n=50,
        # worse than at n=20 -- a symptom of instability, not of "less data
        # is worse," which defeats the point of the curve).
        sub_rank = select_rank_by_energy(sub_data, energy_threshold=0.99)
        sub_rank = min(sub_rank, n - 1)
        m = fit_dmd(sub_data, sub_times, rank=sub_rank)
        full = dmd_forecast(m, sub_data[:, 0], (n + n_test) - 1)
        return full[:, n:]

    (dmd_model, dmd_fit_time) = time_call(fit_dmd, train.data_matrix, train.times, rank)
    full_forecast = dmd_forecast(dmd_model, train.data_matrix[:, 0], n_train + n_test - 1)
    results["DMD"] = {
        "reconstruction_err": relative_l2_error(full_forecast[:, :n_train], train.data_matrix),
        "extrapolation_err": relative_l2_error(full_forecast[:, n_train:], test.data_matrix),
        "fit_time_s": dmd_fit_time,
    }

    # ---- Neural ROM (black-box latent dynamics, Phase 2's "Arm 1") ----
    # Using rollout_horizon=30 and epochs=800 for the transient-including window
    def neural_forecast_with_n(n: int) -> pt.Tensor:
        sub_data, sub_times = train.data_matrix[:, :n], train.times[:n]
        m, _ = fit_neural_rom(sub_data, sub_times, latent_dim=latent_dim, epochs=800, rollout_horizon=30)
        return neural_forecast(m, sub_data[:, -1], test.times)

    (neural_model, neural_history), neural_fit_time = time_call(
        fit_neural_rom, train.data_matrix, train.times, latent_dim, 800, rollout_horizon=30
    )
    neural_train_pred = neural_forecast(neural_model, train.data_matrix[:, 0], train.times)
    neural_test_pred = neural_forecast(neural_model, train.data_matrix[:, -1], test.times)
    results["NeuralROM"] = {
        "reconstruction_err": relative_l2_error(neural_train_pred, train.data_matrix),
        "extrapolation_err": relative_l2_error(neural_test_pred, test.data_matrix),
        "fit_time_s": neural_fit_time,
    }

    print_summary_table(results)

    # ---- Data efficiency: extrapolation error vs. number of training
    # snapshots used, for the two baselines that actually forecast ----
    # CAUTION, found while testing this against the synthetic fixture:
    # energy-threshold rank selection (select_rank_by_energy) can be
    # unstable at small snapshot counts, especially with noisy data --
    # observed non-monotonic DMD extrapolation error across n (e.g. n=20
    # and n=100 both far worse than n=50 and n=210, no clear trend) purely
    # from which rank the 99% threshold happened to land on for that
    # subset, not from a genuine "less data is worse" signal. Two things
    # worth doing before trusting this curve on real data: (1) look at
    # which rank got selected at each n, not just the resulting error, and
    # (2) average over multiple random seeds/subsample choices per n --
    # this runs each n once, and a single DMD fit or neural ROM training
    # run at a given data size is one sample, not a stable estimate (same
    # statistical-hygiene point as Phase 2's multi-seed guidance).
    snapshot_counts = [n for n in [20, 50, 100, n_train] if n <= n_train]
    print(f"\nData efficiency (extrapolation error vs. training snapshots used): {snapshot_counts}")
    dmd_curve = data_efficiency_curve(dmd_forecast_with_n, test.data_matrix, snapshot_counts)
    print(f"  DMD:        {dmd_curve}")
    neural_curve = data_efficiency_curve(neural_forecast_with_n, test.data_matrix, snapshot_counts)
    print(f"  NeuralROM:  {neural_curve}")

    results["data_efficiency"] = {"DMD": dmd_curve, "NeuralROM": neural_curve}

    # ---- Save results ----
    _save_results(snaps, results, train, test, rank, dmd_model,
                  dmd_curve, neural_curve, snapshot_counts, t_split)

    return results


def _save_results(snaps, results, train, test, rank, dmd_model,
                  dmd_curve, neural_curve, snapshot_counts, t_split):
    """Save all result files to the results/ folder."""
    import flowtorch

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- comparison_table.csv ---
    with open(results_dir / "comparison_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["baseline", "reconstruction_err", "extrapolation_err", "fit_time_s"])
        for name in ["POD", "DMD", "NeuralROM"]:
            row = results.get(name, {})
            writer.writerow([
                name,
                row.get("reconstruction_err", ""),
                row.get("extrapolation_err", ""),
                row.get("fit_time_s", ""),
            ])

    # --- data_efficiency_dmd.json ---
    with open(results_dir / "data_efficiency_dmd.json", "w") as f:
        json.dump({
            "metric": "extrapolation_error",
            "description": "DMD extrapolation relative L2 error vs. number of training snapshots used for fitting",
            "snapshot_counts": snapshot_counts,
            "errors": dmd_curve,
            "key_parameters": {
                "t_split": t_split,
                "n_train": len(train.times),
                "n_test": len(test.times),
                "dt": train.dt,
            },
        }, f, indent=2)

    # --- data_efficiency_neuralrom.json ---
    with open(results_dir / "data_efficiency_neuralrom.json", "w") as f:
        json.dump({
            "metric": "extrapolation_error",
            "description": "NeuralROM extrapolation relative L2 error vs. number of training snapshots used for fitting",
            "snapshot_counts": snapshot_counts,
            "errors": neural_curve,
            "key_parameters": {
                "t_split": t_split,
                "n_train": len(train.times),
                "n_test": len(test.times),
                "dt": train.dt,
                "latent_dim": 8,
                "rollout_horizon": 30,
                "epochs": 800,
            },
        }, f, indent=2)

    # --- pod_reconstruction_vs_rank.csv ---
    # Compute POD reconstruction error for a range of ranks
    ranks_to_test = sorted({1, 2, 4, min(rank, train.data_matrix.shape[1] - 1)})
    pod_errs = reconstruction_error_vs_rank(train.data_matrix, ranks_to_test)
    with open(results_dir / "pod_reconstruction_vs_rank.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "relative_reconstruction_error"])
        for r in sorted(pod_errs.keys()):
            writer.writerow([r, pod_errs[r]])

    # --- dmd_eigenvalues.csv ---
    eigvals = dmd_model.dmd.eigvals
    dt = train.dt
    eig_rows = []
    for lam in eigvals:
        magnitude = abs(lam).item()
        # continuous-time growth rate: ln(|λ|)/dt
        growth_rate = float(pt.log(pt.tensor(magnitude)) / dt) if magnitude > 0 else float("-inf")
        # continuous-time frequency in Hz: angle(λ)/(2π*dt)
        angle = pt.angle(lam).item()
        frequency_hz = angle / (2 * pt.pi * dt) if dt > 0 else float("nan")
        eig_rows.append({
            "magnitude": magnitude,
            "growth_rate": growth_rate,
            "frequency_hz": frequency_hz,
        })
    with open(results_dir / "dmd_eigenvalues.json", "w") as f:
        json.dump({
            "metric": "dmd_eigenvalues",
            "description": "DMD eigenvalues with magnitude, continuous-time growth rate, and frequency in Hz",
            "eigenvalues": eig_rows,
            "key_parameters": {
                "dt": dt,
                "n_train": len(train.times),
                "rank_used": rank,
            },
        }, f, indent=2)

    # --- pod_rank_selection.json ---
    # Rank selected at each training size n=20, 50, 100, 160
    training_sizes = [20, 50, 100, min(160, len(train.times))]
    rank_selection = {}
    for n in training_sizes:
        sub_data = train.data_matrix[:, :n]
        selected_rank = select_rank_by_energy(sub_data, energy_threshold=0.99)
        selected_rank = min(selected_rank, n - 1)
        rank_selection[str(n)] = {"rank_selected": selected_rank}
    with open(results_dir / "pod_rank_selection.json", "w") as f:
        json.dump({
            "metric": "rank_selection",
            "description": "DMD rank selected by 99% cumulative energy at each training snapshot count",
            "training_sizes": training_sizes,
            "rank_selection": rank_selection,
            "key_parameters": {
                "energy_threshold": 0.99,
                "t_split": t_split,
            },
        }, f, indent=2)

    # --- full_results.pkl ---
    with open(results_dir / "full_results.pkl", "wb") as f:
        pickle.dump(results, f)

    # --- run_metadata.json ---
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "dataset_path": str(flowtorch.DATASETS["of_cylinder2D_binary"] if "of_cylinder2D_binary" in flowtorch.DATASETS else "not set"),
        "t_split": t_split,
        "dt": train.dt,
        "n_train": len(train.times),
        "n_test": len(test.times),
        "rank_99": rank,
        "latent_dim": 8,
        "neuralrom_epochs": 800,
        "neuralrom_rollout_horizon": 30,
    }
    with open(results_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    snaps = load_cylinder_snapshots()   # real data -- requires FLOWTORCH_DATASETS set up
    # default window (t_min=None) includes the transient: first non-zero time
    # step (t=0.025) through t=10.0, 400 snapshots. t_split=8.0 gives ~320
    # training / ~80 test snapshots from that window -- adjust once you've
    # looked at the actual data; this default is a starting point, not a
    # tuned choice. (Pass load_cylinder_snapshots(t_min=4.0) to reproduce
    # the old 241-snapshot post-transient window.)
    run(snaps, t_split=8.0)
