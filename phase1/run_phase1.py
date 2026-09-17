"""
Phase 1 orchestration. Loads the real cylinder2D dataset (see
data_loading.py for setup), splits into a training window and a held-out
extrapolation window, fits POD / DMD / the neural ROM baseline, and prints
the comparison table across the four metrics.

Default run is POD + DMD only (NeuralROM is parked -- pass --with-neural to
re-enable it). DMD is additionally swept over a fixed-rank ladder
[1, 2, 4, 8, 15, 32, rank_99] to locate where the rank-63 blowup onsets vs.
the stable-but-mediocre low-rank regime (see status.md §8.4 / future.md).

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
from dmd_baseline import fit_dmd, forecast as dmd_forecast, eigen_summary
from neural_rom_baseline import fit_neural_rom, forecast as neural_forecast
from metrics import (
    relative_l2_error, time_call, data_efficiency_curve, print_summary_table,
    fluctuation_relative_error, mean_predictor, persistence_predictor, pod_projection_floor,
)

SWEEP_RANKS_BASE = (1, 2, 4, 8, 15, 32)


def run(snaps: CylinderSnapshots, t_split: float, latent_dim: int = 8,
        with_neural: bool = False, output_dir: str = "results") -> dict:
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

    # ---- DMD rank sweep: extrapolation error vs. fixed rank ----
    # Fixed-rank fits (no energy selection) spanning below/above the Expt-1
    # rank (15) up to this window's energy-selected rank_99 (63). Locates
    # where the canonical rank-63 blowup onsets vs. the stable but mediocre
    # low-rank regime (recon 0.27 / extrap 0.30 at rank 15).
    sweep_ranks = sorted(set([*SWEEP_RANKS_BASE, rank]))
    dmd_sweep: dict = {}
    print(f"\nDMD rank sweep (fixed rank, no energy selection): {sweep_ranks}")
    for r in sweep_ranks:
        (m, fit_time) = time_call(fit_dmd, train.data_matrix, train.times, r)
        full = dmd_forecast(m, train.data_matrix[:, 0], n_train + n_test - 1)
        dmd_sweep[str(r)] = {
            "rank": r,
            "reconstruction_err": relative_l2_error(full[:, :n_train], train.data_matrix),
            "extrapolation_err": relative_l2_error(full[:, n_train:], test.data_matrix),
            "fit_time_s": fit_time,
            "eigen_summary": eigen_summary(m),
        }
        print(f"  rank={r:>2}  recon={dmd_sweep[str(r)]['reconstruction_err']:.6e}  "
              f"extrap={dmd_sweep[str(r)]['extrapolation_err']:.6e}  "
              f"n_unstable={dmd_sweep[str(r)]['eigen_summary']['n_unstable']}")
    results["dmd_rank_sweep"] = dmd_sweep

    # ---- Neural ROM (black-box latent dynamics, Phase 2's "Arm 1") ----
    # PARKED (2026-09-06): oversized-for-data black-box with no physical
    # hard constraint; inferior to POD/DMD. Code kept, gated behind
    # --with-neural so it can be re-enabled without surgery.
    if with_neural:
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
    else:
        neural_curve = None

    print_summary_table({k: results[k] for k in ("POD", "DMD", "NeuralROM") if k in results})

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
    if with_neural:
        neural_curve = data_efficiency_curve(neural_forecast_with_n, test.data_matrix, snapshot_counts)
        print(f"  NeuralROM:  {neural_curve}")
        results["data_efficiency"] = {"DMD": dmd_curve, "NeuralROM": neural_curve}
    else:
        results["data_efficiency"] = {"DMD": dmd_curve}

    # ---- Save results ----
    _save_results(snaps, results, train, test, rank, dmd_model,
                  dmd_curve, neural_curve, snapshot_counts, t_split,
                  dmd_sweep, output_dir, with_neural)

    return results


def _save_results(snaps, results, train, test, rank, dmd_model,
                  dmd_curve, neural_curve, snapshot_counts, t_split,
                  dmd_sweep, output_dir="results", with_neural=False):
    """Save all result files to the given output folder (default results/)."""
    import flowtorch

    results_dir = Path(__file__).parent / output_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- comparison_table.csv ---
    with open(results_dir / "comparison_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["baseline", "reconstruction_err", "extrapolation_err", "fit_time_s"])
        baseline_names = ["POD", "DMD"] + (["NeuralROM"] if with_neural else [])
        for name in baseline_names:
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

    # --- data_efficiency_neuralrom.json (only when NeuralROM enabled) ---
    if with_neural:
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

    # --- dmd_error_vs_rank.csv ---
    # Fixed-rank sweep: error vs. rank over the same transient window.
    sweep_ranks = sorted(int(k) for k in dmd_sweep.keys())
    with open(results_dir / "dmd_error_vs_rank.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "reconstruction_err", "extrapolation_err", "fit_time_s"])
        for r in sweep_ranks:
            row = dmd_sweep[str(r)]
            writer.writerow([r, row["reconstruction_err"], row["extrapolation_err"], row["fit_time_s"]])

    # --- dmd_rank_sweep.json ---
    # Per-rank errors plus eigenvalue diagnostics (n_unstable, dominant
    # unstable modes) -- the artifact that locates where the rank-63 blowup
    # onsets vs. the stable-but-mediocre low-rank regime.
    sweep_json = {
        "metric": "dmd_rank_sweep",
        "description": "Fixed-rank DMD errors and eigenvalue diagnostics on the transient window, no energy selection",
        "ranks": sweep_ranks,
        "per_rank": {
            r: {k: v for k, v in dmd_sweep[str(r)].items() if k != "rank"}
            for r in sweep_ranks
        },
        "key_parameters": {
            "t_split": t_split,
            "n_train": len(train.times),
            "n_test": len(test.times),
            "dt": train.dt,
            "rank_99": rank,
        },
    }
    with open(results_dir / "dmd_rank_sweep.json", "w") as f:
        json.dump(sweep_json, f, indent=2)

    # --- pod_rank_selection.json ---
    # Rank selected at each training size n=20, 50, 100, n_train -- kept in
    # sync with the data-efficiency curve above (was min(160, ...) before,
    # which mislabeled the n_train point as "160").
    training_sizes = [20, 50, 100, len(train.times)]
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
        "dmd_rank_sweep": sweep_ranks,
        "with_neural": with_neural,
        "output_dir": output_dir,
    }
    if with_neural:
        metadata.update({
            "latent_dim": 8,
            "neuralrom_epochs": 800,
            "neuralrom_rollout_horizon": 30,
        })
    with open(results_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


ROLLING_H = 20
ROLLING_ORIGINS = (0, 20, 40)   # indices into test.data_matrix's columns
DATA_EFF_COUNTS = (20, 50, 100, 160, 319)
DATA_EFF_FIXED_RANKS = (8, 15, 32)


def _forecast_real(model, initial_condition: pt.Tensor, n_steps: int) -> tuple[pt.Tensor, float]:
    """Wraps dmd_forecast, dropping the imaginary part (DMD.predict on real
    data can return a complex tensor with a tiny imaginary component from
    complex-conjugate eigenvalue pairs) and returning a sanity ratio
    max|imag|/max|real| so a real numerical problem (large imaginary part)
    isn't silently swallowed by `.real`."""
    raw = dmd_forecast(model, initial_condition, n_steps)
    if pt.is_complex(raw):
        real = raw.real
        max_real = real.abs().max().item()
        max_imag = raw.imag.abs().max().item()
        ratio = max_imag / max_real if max_real > 0 else float("nan")
        return real, ratio
    return raw, 0.0


def _errors(pred: pt.Tensor, true: pt.Tensor, reference_mean: pt.Tensor) -> dict:
    return {
        "full_field": relative_l2_error(pred, true),
        "fluctuation": fluctuation_relative_error(pred, true, reference_mean),
    }


def _rolling_origin_list(last_train_snapshot: pt.Tensor, test: CylinderSnapshots,
                          H: int = ROLLING_H, test_origins: tuple = ROLLING_ORIGINS) -> list[tuple[str, pt.Tensor, pt.Tensor]]:
    """Origins = the last training snapshot, plus true test snapshots at
    test_origins -- per spec, "last training snapshot plus true test
    snapshots at test indices 0, 20, 40". The last-train origin's target is
    test[:, 0:H] (its forecast overlaps the start of the primary 81-step
    forecast, which is expected -- it is the same restart point at a
    shorter, fixed horizon)."""
    origins = [("last_train", last_train_snapshot, test.data_matrix[:, 0:H])]
    for idx in test_origins:
        origins.append((f"test_idx_{idx}", test.data_matrix[:, idx], test.data_matrix[:, idx + 1: idx + 1 + H]))
    return origins


def _rolling_origin_errors(model, last_train_snapshot: pt.Tensor, test: CylinderSnapshots,
                            reference_mean: pt.Tensor, H: int = ROLLING_H,
                            test_origins: tuple = ROLLING_ORIGINS) -> dict:
    labels, full_errs, fluct_errs = [], [], []
    for label, origin_state, true_window in _rolling_origin_list(last_train_snapshot, test, H, test_origins):
        pred, _ = _forecast_real(model, origin_state, H)
        pred_window = pred[:, 1:]
        e = _errors(pred_window, true_window, reference_mean)
        labels.append(label)
        full_errs.append(e["full_field"])
        fluct_errs.append(e["fluctuation"])
    return {
        "origin_labels": labels,
        "per_origin_full_field": full_errs,
        "per_origin_fluctuation": fluct_errs,
        "mean_full_field": sum(full_errs) / len(full_errs),
        "max_full_field": max(full_errs),
        "mean_fluctuation": sum(fluct_errs) / len(fluct_errs),
        "max_fluctuation": max(fluct_errs),
    }


def _window_sweep_ranks(sweep_ranks: list[int], n_fit: int) -> list[int]:
    return sorted({r for r in sweep_ranks if r <= n_fit - 1})


def run_v2(snaps: CylinderSnapshots, t_split: float = 8.0, output_dir: str = "results_v2") -> dict:
    """Corrected Phase 1 evaluation protocol -- see PATH_FORWARD.md Step 1.
    Every DMD forecast restarts from the LAST training snapshot (not t=0),
    always compared against three reference predictors (mean, persistence,
    POD(train) projection floor), always reported as both full-field
    relative L2 and fluctuation-normalised error, and always accompanied by
    eigenvalue diagnostics. The legacy from-x0 rollout is kept as a single
    clearly-labelled provenance column, not the primary score."""
    assert snaps.data_matrix.dtype == pt.float64, (
        f"run_v2 requires float64 input (float32 lets flowTorch's Gram-matrix "
        f"SVD destroy mode orthogonality at high rank, see "
        f"data_loading.py's load_cylinder_snapshots docstring and "
        f"results_v2/precision_check.json); got {snaps.data_matrix.dtype}. "
        f"Pass load_cylinder_snapshots() with its float64 default, don't override dtype."
    )
    train, test = snaps.split(t_split)
    n_train, n_test = len(train.times), len(test.times)
    print(f"[v2] Train: {n_train} snapshots (t < {t_split}s)   Test: {n_test} snapshots (t >= {t_split}s)")

    train_mean = train.data_matrix.mean(dim=1)
    last_train_snapshot = train.data_matrix[:, -1]

    # Fit windows: full training window, and the post-transient sub-window
    # 4.0 <= t < t_split (both end at the same last training snapshot, so
    # forecasts restarted from "the last training snapshot" are directly
    # comparable across windows).
    _before_4, train_post_transient = train.split(4.0)   # times >= 4.0
    windows = {"full": train, "post_transient": train_post_transient}

    r99_full = select_rank_by_energy(train.data_matrix, 0.99)
    r999_full = select_rank_by_energy(train.data_matrix, 0.999)
    sweep_ranks_base = sorted(set([*SWEEP_RANKS_BASE, 63, r99_full, r999_full]))
    print(f"[v2] Full-window energy ranks (s**2): 99%={r99_full}  99.9%={r999_full}")
    print(f"[v2] Sweep rank ladder (pre-cap): {sweep_ranks_base}")

    window_energy_ranks = {}
    sweep_rows: list[dict] = []
    r_convention_checks = []
    for window_name, w in windows.items():
        n_fit = len(w.times)
        # NOTE on rank convention: select_rank_by_energy computes energy on
        # the MEAN-CENTRED data, but DMD is fit on w.data_matrix RAW
        # (uncentred) -- its leading singular vector is ~the mean flow, not
        # a fluctuation mode. So "DMD rank r" may carry only r-1 genuine
        # fluctuation modes, and odd/even parity can split a conjugate
        # pair. r99_w+1 / r999_w+1 are added below to check whether the one
        # extra dimension (completing a pair, or picking up the mean-flow
        # direction the centred energy count doesn't see) changes the
        # result materially -- fitting itself is unchanged.
        r99_w = select_rank_by_energy(w.data_matrix, 0.99)
        r999_w = select_rank_by_energy(w.data_matrix, 0.999)
        window_energy_ranks[window_name] = {"n_fit": n_fit, "r99": r99_w, "r999": r999_w}
        extra_ranks = {
            min(r99_w, n_fit - 1), min(r99_w + 1, n_fit - 1),
            min(r999_w, n_fit - 1), min(r999_w + 1, n_fit - 1),
        }
        ranks_w = sorted(set(_window_sweep_ranks(sweep_ranks_base, n_fit)) | extra_ranks)
        print(f"\n[v2] Window={window_name}  n_fit={n_fit}  ranks={ranks_w}  (r99+1={r99_w + 1}, r999+1={r999_w + 1})")

        # ---- window-specific POD projection floor: the POD basis is
        # fitted on THIS window's own snapshots (not the full-window basis)
        # -- the post-transient DMD fit can legitimately beat the
        # full-window floor since it operates in a different, narrower
        # basis; comparing it to the full-window floor was an apples-to-
        # oranges mistake. ----
        pod_w = fit_pod(w.data_matrix, rank=max(ranks_w))

        for r in ranks_w:
            (model, fit_time) = time_call(fit_dmd, w.data_matrix, w.times, r)
            eig = eigen_summary(model)
            max_lambda = max((row["magnitude"] for row in eig["eigenvalues"]), default=float("nan"))

            # (a) primary: restart from last training snapshot, forecast the
            # full 81-step test window.
            pred_full, imag_ratio = _forecast_real(model, last_train_snapshot, n_test)
            pred_full = pred_full[:, 1:]
            primary = _errors(pred_full, test.data_matrix, train_mean)

            # (b) rolling origins, fixed horizon H: last training snapshot,
            # plus TRUE test snapshots at indices 0, 20, 40.
            rolling = _rolling_origin_errors(model, last_train_snapshot, test, train_mean)

            # (c) legacy: from-x0 rollout over this window's own training +
            # test span, extrapolation portion only -- provenance column,
            # not a primary score (see module docstring).
            legacy_pred, legacy_imag_ratio = _forecast_real(model, w.data_matrix[:, 0], n_fit + n_test - 1)
            legacy_extrap = _errors(legacy_pred[:, n_fit:], test.data_matrix, train_mean)

            # window-specific POD(train window) projection floor: same rank
            # r, but the basis comes from THIS window's own snapshots.
            # Fluctuation normalisation still uses the full training mean
            # (train_mean), for comparability across windows/rows; only the
            # projection basis and centring are window-specific.
            pod_floor_pred = pod_projection_floor(pod_w.modes, pod_w.mean, test.data_matrix, r)
            pod_floor = _errors(pod_floor_pred, test.data_matrix, train_mean)
            ratio = (primary["full_field"] / pod_floor["full_field"]) if pod_floor["full_field"] > 0 else float("inf")

            row = {
                "window": window_name,
                "rank": r,
                "n_fit": n_fit,
                "fit_time_s": fit_time,
                "n_unstable": eig["n_unstable"],
                "max_lambda": max_lambda,
                "max_imag_over_real": imag_ratio,
                "primary_full_field": primary["full_field"],
                "primary_fluctuation": primary["fluctuation"],
                "pod_floor_full_field": pod_floor["full_field"],
                "pod_floor_fluctuation": pod_floor["fluctuation"],
                "ratio": ratio,
                "rolling_mean_full_field": rolling["mean_full_field"],
                "rolling_max_full_field": rolling["max_full_field"],
                "rolling_mean_fluctuation": rolling["mean_fluctuation"],
                "rolling_max_fluctuation": rolling["max_fluctuation"],
                "legacy_extrap_full_field": legacy_extrap["full_field"],
                "legacy_extrap_fluctuation": legacy_extrap["fluctuation"],
                "legacy_max_imag_over_real": legacy_imag_ratio,
                "eigen_summary": eig,
            }
            sweep_rows.append(row)
            print(f"  rank={r:>3}  primary(full/fluct)={primary['full_field']:.4e}/{primary['fluctuation']:.4e}  "
                  f"pod_floor(full/fluct)={pod_floor['full_field']:.4e}/{pod_floor['fluctuation']:.4e}  "
                  f"ratio={ratio:.3f}  "
                  f"rolling_mean(full/fluct)={rolling['mean_full_field']:.4e}/{rolling['mean_fluctuation']:.4e}  "
                  f"legacy_extrap={legacy_extrap['full_field']:.4e}  n_unstable={eig['n_unstable']}  "
                  f"max|lambda|={max_lambda:.4f}")

        # ---- r vs r+1 comparison for the energy-rank-convention check ----
        for label, r_base in (("r99", r99_w), ("r999", r999_w)):
            r_plus1 = min(r_base + 1, n_fit - 1)
            row_base = next((r for r in sweep_rows if r["window"] == window_name and r["rank"] == r_base), None)
            row_plus1 = next((r for r in sweep_rows if r["window"] == window_name and r["rank"] == r_plus1), None)
            if row_base is not None and row_plus1 is not None and r_base != r_plus1:
                delta = row_plus1["primary_full_field"] - row_base["primary_full_field"]
                rel_delta = delta / row_base["primary_full_field"] if row_base["primary_full_field"] != 0 else float("nan")
                r_convention_checks.append({
                    "window": window_name, "which": label,
                    "r": r_base, "r_plus1": r_plus1,
                    "primary_full_field_at_r": row_base["primary_full_field"],
                    "primary_full_field_at_r_plus1": row_plus1["primary_full_field"],
                    "relative_change": rel_delta,
                })
                print(f"  [{window_name}] {label}: r={r_base} err={row_base['primary_full_field']:.4e}  "
                      f"r+1={r_plus1} err={row_plus1['primary_full_field']:.4e}  "
                      f"relative_change={rel_delta:+.3f}")

    # ---- Reference predictors (computed once, on the full window) ----
    mean_pred_full = mean_predictor(train_mean, n_test)
    persistence_pred_full = persistence_predictor(last_train_snapshot, n_test)
    mean_ref = _errors(mean_pred_full, test.data_matrix, train_mean)
    persistence_ref = _errors(persistence_pred_full, test.data_matrix, train_mean)

    persistence_rolling_labels, persistence_rolling_full, persistence_rolling_fluct = [], [], []
    for label, origin_state, true_window in _rolling_origin_list(last_train_snapshot, test):
        pred = persistence_predictor(origin_state, ROLLING_H)
        e = _errors(pred, true_window, train_mean)
        persistence_rolling_labels.append(label)
        persistence_rolling_full.append(e["full_field"])
        persistence_rolling_fluct.append(e["fluctuation"])
    persistence_rolling = {
        "origin_labels": persistence_rolling_labels,
        "per_origin_full_field": persistence_rolling_full,
        "per_origin_fluctuation": persistence_rolling_fluct,
        "mean_full_field": sum(persistence_rolling_full) / len(persistence_rolling_full),
        "max_full_field": max(persistence_rolling_full),
        "mean_fluctuation": sum(persistence_rolling_fluct) / len(persistence_rolling_fluct),
        "max_fluctuation": max(persistence_rolling_fluct),
    }

    # POD projection floor, per window (each window's basis fitted on that
    # window's own snapshots) -- pulled straight from the sweep rows so this
    # is exactly the same floor each sweep row's "ratio" column was computed
    # against, not a separately recomputed number.
    pod_projection_floor_by_window: dict = {}
    for window_name in windows:
        pod_projection_floor_by_window[window_name] = {
            str(row["rank"]): {"full_field": row["pod_floor_full_field"], "fluctuation": row["pod_floor_fluctuation"]}
            for row in sweep_rows if row["window"] == window_name
        }

    reference_predictors = {
        "mean": mean_ref,
        "persistence": persistence_ref,
        "persistence_rolling": persistence_rolling,
        "pod_projection_floor_by_window": pod_projection_floor_by_window,
    }
    print(f"\n[v2] Reference predictors: mean(full/fluct)={mean_ref['full_field']:.4f}/{mean_ref['fluctuation']:.4f}  "
          f"persistence(full/fluct)={persistence_ref['full_field']:.4f}/{persistence_ref['fluctuation']:.4f}")

    # ---- Data efficiency: fit on the LAST n training snapshots ----
    data_efficiency = {}
    for n in DATA_EFF_COUNTS:
        if n > n_train:
            continue
        sub_data = train.data_matrix[:, -n:]
        sub_times = train.times[-n:]
        r99_sub = select_rank_by_energy(sub_data, 0.99)
        ranks_n = sorted({min(r, n - 1) for r in (*DATA_EFF_FIXED_RANKS, r99_sub)})
        per_rank = {}
        for r in ranks_n:
            try:
                (model, fit_time) = time_call(fit_dmd, sub_data, sub_times, r)
                pred, imag_ratio = _forecast_real(model, sub_data[:, -1], n_test)
                e = _errors(pred[:, 1:], test.data_matrix, train_mean)
                per_rank[str(r)] = {**e, "fit_time_s": fit_time, "max_imag_over_real": imag_ratio}
            except RuntimeError as exc:
                # Near-full-rank fits on very short subsets (e.g. n=20,
                # rank capped to n-1=19) can produce a numerically singular
                # companion matrix -- record the failure instead of crashing
                # the whole sweep; this is itself a data-efficiency finding
                # (fitting near-full-rank DMD off very few snapshots is not
                # just inaccurate, it can be numerically unstable).
                per_rank[str(r)] = {"error": str(exc)}
                print(f"    [n={n} rank={r}] DMD fit failed: {exc}")
        data_efficiency[str(n)] = {"r99_rank": r99_sub, "ranks": ranks_n, "per_rank": per_rank}
        print(f"[v2] Data efficiency n={n:>3}  r99={r99_sub}  ranks={ranks_n}")

    # ---- Write outputs ----
    results_dir = Path(__file__).parent / output_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    sweep_fieldnames = [
        "window", "rank", "n_fit", "fit_time_s", "n_unstable", "max_lambda", "max_imag_over_real",
        "primary_full_field", "primary_fluctuation",
        "pod_floor_full_field", "pod_floor_fluctuation", "ratio",
        "rolling_mean_full_field", "rolling_max_full_field",
        "rolling_mean_fluctuation", "rolling_max_fluctuation",
        "legacy_extrap_full_field", "legacy_extrap_fluctuation", "legacy_max_imag_over_real",
    ]
    with open(results_dir / "sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sweep_fieldnames)
        writer.writeheader()
        for row in sweep_rows:
            writer.writerow({k: row[k] for k in sweep_fieldnames})

    with open(results_dir / "sweep.json", "w") as f:
        json.dump({
            "description": "Rank x fit-window DMD sweep, restart-from-last-train-snapshot protocol. "
                            "pod_floor_* is the projection floor of a POD basis fitted on THAT ROW's "
                            "own fit window (not the full-window basis); ratio = primary_full_field / "
                            "pod_floor_full_field.",
            "rows": sweep_rows,
            "window_energy_ranks": window_energy_ranks,
            "rank_convention_checks": r_convention_checks,
        }, f, indent=2)

    with open(results_dir / "reference_predictors.json", "w") as f:
        json.dump(reference_predictors, f, indent=2)

    with open(results_dir / "data_efficiency.json", "w") as f:
        json.dump(data_efficiency, f, indent=2)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "protocol": (
            "Primary: DMD forecast restarted from the last training snapshot, "
            "covering the full held-out test window (n_test steps). Rolling "
            "origins: fixed horizon H, origins = the last training snapshot "
            "plus true test snapshots at the given test indices. Legacy: "
            "from-x0 rollout extrapolation error (provenance only). All "
            "errors reported as full-field relative L2 and "
            "fluctuation-normalised (relative to the training temporal "
            "mean). POD projection floor is window-specific: the basis is "
            "fitted on the same fit window as the DMD row it's compared "
            "against, not a single full-window basis."
        ),
        "dtype": str(train.data_matrix.dtype),
        "t_split": t_split,
        "n_train": n_train,
        "n_test": n_test,
        "dt": train.dt,
        "rolling_horizon_H": ROLLING_H,
        "rolling_origins": ["last_train", *[f"test_idx_{i}" for i in ROLLING_ORIGINS]],
        "sweep_rank_ladder": sweep_ranks_base,
        "windows": list(windows.keys()),
        "window_energy_ranks": window_energy_ranks,
        "rank_convention_checks": r_convention_checks,
        "data_efficiency_counts": [n for n in DATA_EFF_COUNTS if n <= n_train],
        "data_efficiency_fixed_ranks": list(DATA_EFF_FIXED_RANKS),
    }
    with open(results_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[v2] Wrote results to {results_dir}")

    return {
        "sweep": sweep_rows,
        "reference_predictors": reference_predictors,
        "data_efficiency": data_efficiency,
        "window_energy_ranks": window_energy_ranks,
        "rank_convention_checks": r_convention_checks,
        "metadata": metadata,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 1 baselines on the real cylinder2D transient window: POD + DMD (rank sweep).",
    )
    parser.add_argument("--with-neural", action="store_true",
                        help="also fit the NeuralROM baseline (default OFF -- parked 2026-09-06, "
                             "not part of the corrected protocol -- do not run)")
    parser.add_argument("--output-dir", default="results_v2",
                        help="output folder under phase1/ (default: results_v2)")
    parser.add_argument("--legacy", action="store_true",
                        help="also run the old (flawed) protocol into --output-dir (default off)")
    args = parser.parse_args()

    snaps = load_cylinder_snapshots()   # real data -- requires FLOWTORCH_DATASETS set up
    # default window (t_min=None) includes the transient: first non-zero time
    # step (t=0.025) through t=10.0, 400 snapshots. t_split=8.0 gives ~320
    # training / ~80 test snapshots from that window -- adjust once you've
    # looked at the actual data; this default is a starting point, not a
    # tuned choice. (Pass load_cylinder_snapshots(t_min=4.0) to reproduce
    # the old 241-snapshot post-transient window.)
    if args.legacy:
        run(snaps, t_split=8.0, with_neural=args.with_neural, output_dir=args.output_dir)
    else:
        run_v2(snaps, t_split=8.0, output_dir=args.output_dir)
