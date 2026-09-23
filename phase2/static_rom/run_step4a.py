"""
Step 4a orchestration: static (closed-form) OpInf null control on POD
coefficients of the real cylinder2D dataset. See PATH_FORWARD.md Step 4a
specification for the full protocol; this module wires together
case_activity.py (which terms are active), term_library.py (Arm 2 textbook
vs. Arm 3 parsed regressor families), and opinf.py (the fit/forecast
engine) into the grid x sanity-check x noise-band run described there.

Run as a module from phase2/:
    python -m static_rom.run_step4a
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch as pt

_PHASE2_DIR = Path(__file__).resolve().parent.parent
_PHASE1_DIR = _PHASE2_DIR.parent / "phase1"
if str(_PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASE1_DIR))

# Phase 1 modules, imported directly (NOT run_phase1, which pulls in the
# parked NeuralROM baseline -- see PATH_FORWARD.md Step 4a plan).
from data_loading import load_cylinder_snapshots, CylinderSnapshots      # noqa: E402
from pod_baseline import fit_pod                                          # noqa: E402
from dmd_baseline import fit_dmd, forecast as dmd_forecast                # noqa: E402
from metrics import relative_l2_error, fluctuation_relative_error, pod_projection_floor  # noqa: E402

from extractor.operator_graph import build_operator_graph                 # noqa: E402
from static_rom.case_activity import read_case_activity                  # noqa: E402
from static_rom.term_library import arm2_library, arm3_library           # noqa: E402
from static_rom.opinf import (                                            # noqa: E402
    LAMBDA_GRID, Library, build_regressors, build_target, fit_ridge,
    pod_basis, project, reconstruct, rollout, select_lambda,
)

RANKS = (8, 11, 15, 23)
TARGETS = ("discrete", "continuous")
N_TEST = 81
T_SPLIT = 8.0
VAL_FRACTION = 0.2
ROLLOUT_STEPS_FOR_SELECTION = 20
N_LOO_BLOCKS = 5

DEFAULT_CASE_DIR = str((_PHASE2_DIR.parent / "data" / "datasets_29_10_2021" / "datasets" / "of_cylinder2D_binary"))
DEFAULT_SOURCE_DIR = str(_PHASE2_DIR / "fixtures" / "pimpleFoam_v2006")
DEFAULT_OUT = "results_step4a"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_snapshots(data_dir: str | None) -> CylinderSnapshots:
    """Thin generalisation of data_loading.load_cylinder_snapshots (which
    can only load the flowTorch-registered 'of_cylinder2D_binary' dataset)
    for an arbitrary OpenFOAM case directory -- e.g. Step 4b's modified-
    solver case, once it exists. Same masking/window defaults; do not
    modify phase1/data_loading.py itself (out of scope for Step 4a)."""
    if data_dir is None:
        return load_cylinder_snapshots()
    from flowtorch.data import FOAMDataloader, mask_box

    domain_lower, domain_upper = (0.1, -1.0), (0.75, 1.0)
    loader = FOAMDataloader(data_dir)
    all_times = loader.write_times
    window_times = [t for t in all_times if float(t) != 0.0]
    dt = float(window_times[1]) - float(window_times[0])
    vertices = loader.vertices[:, :2]
    mask = mask_box(vertices, lower=list(domain_lower), upper=list(domain_upper))
    n_selected = int(mask.sum().item())
    data_matrix = pt.zeros((2 * n_selected, len(window_times)), dtype=pt.float64)
    for i, t in enumerate(window_times):
        u = loader.load_snapshot("U", t)
        data_matrix[:n_selected, i] = pt.masked_select(u[:, 0], mask)
        data_matrix[n_selected:, i] = pt.masked_select(u[:, 1], mask)
    return CylinderSnapshots(
        data_matrix=data_matrix, times=[float(t) for t in window_times],
        n_cells_selected=n_selected, mask=mask, vertices=vertices[mask], dt=dt,
    )


def _load_dmd_sweep_reference(path: str) -> dict[tuple[str, int], dict]:
    ref: dict[tuple[str, int], dict] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["window"], int(row["rank"]))
            ref[key] = {
                "primary_full_field": float(row["primary_full_field"]),
                "primary_fluctuation": float(row["primary_fluctuation"]),
                "pod_floor_full_field": float(row["pod_floor_full_field"]),
                "pod_floor_fluctuation": float(row["pod_floor_fluctuation"]),
            }
    return ref


# ---------------------------------------------------------------------------
# Small local helpers
# ---------------------------------------------------------------------------

def _rel_l2(pred: pt.Tensor, true: pt.Tensor) -> float:
    denom = pt.linalg.norm(true)
    return float("inf") if denom == 0 else (pt.linalg.norm(pred - true) / denom).item()


def _errors(pred: pt.Tensor, true: pt.Tensor, reference_mean: pt.Tensor) -> dict:
    return {
        "full_field": relative_l2_error(pred, true),
        "fluctuation": fluctuation_relative_error(pred, true, reference_mean),
    }


def _train_val_split(D: pt.Tensor, Y: pt.Tensor, val_fraction: float = VAL_FRACTION):
    n = D.shape[1]
    n_val = max(1, int(round(n * val_fraction)))
    n_train = n - n_val
    return (D[:, :n_train], Y[:, :n_train]), (D[:, n_train:], Y[:, n_train:])


def _contiguous_blocks(n: int, n_blocks: int = N_LOO_BLOCKS) -> list[tuple[int, int]]:
    edges = [round(i * n / n_blocks) for i in range(n_blocks + 1)]
    return [(edges[i], edges[i + 1]) for i in range(n_blocks) if edges[i + 1] > edges[i]]


def _forecast_from(coef: pt.Tensor, z0: pt.Tensor, kind: str, library: Library,
                    basis, n_test: int, dt: float, max_norm: float,
                    test_data: pt.Tensor, ref_mean: pt.Tensor) -> dict:
    traj, diverged, div_step = rollout(coef, z0, n_test, kind, library, dt=dt, max_norm=max_norm)
    pred = reconstruct(basis, traj[:, 1:])
    if diverged or not pt.isfinite(pred).all():
        return {"full_field": float("inf"), "fluctuation": float("inf"),
                "diverged": True, "divergence_step": div_step}
    e = _errors(pred, test_data, ref_mean)
    e.update(diverged=False, divergence_step=None)
    return e


# ---------------------------------------------------------------------------
# Main grid
# ---------------------------------------------------------------------------

def run_step4a(case_dir: str = DEFAULT_CASE_DIR, source_dir: str = DEFAULT_SOURCE_DIR,
               data_dir: str | None = None, out_dir: str = DEFAULT_OUT) -> dict:
    t_run_start = time.perf_counter()
    timings: dict[str, float] = {}

    results_dir = _PHASE2_DIR / out_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Case activity + libraries ----
    t0 = time.perf_counter()
    case_activity = read_case_activity(case_dir)
    graph = build_operator_graph(
        os.path.join(source_dir, "UEqn.H"),
        fvschemes_path=os.path.join(source_dir, "fvSchemes_of_cylinder2D"),
        extra_sources=[os.path.join(source_dir, "pEqn.H")],
        include_unqualified=True, expand_dispatch=True,
        turbulence_properties_path=os.path.join(source_dir, "turbulenceProperties_of_cylinder2D"),
    )
    lib2 = arm2_library()
    lib3 = arm3_library(graph, equation="UEqn", case_dir=case_dir, case_activity=case_activity)
    timings["build_libraries_s"] = time.perf_counter() - t0
    print(f"[4a] Arm2 families={lib2.families}  Arm3 families={lib3.families}  "
          f"(case_activity: laminar={case_activity.simulation_type == 'laminar'}, "
          f"mrf_active={case_activity.mrf_active}, fvoptions_active={case_activity.fvoptions_active})")

    with open(results_dir / "library_provenance.json", "w") as f:
        json.dump({
            "case_dir": case_dir, "source_dir": source_dir,
            "case_activity": vars(case_activity),
            "arm2": {"families": lib2.families, "provenance": lib2.provenance},
            "arm3": {"families": lib3.families, "provenance": lib3.provenance},
        }, f, indent=2)

    # ---- Data ----
    t0 = time.perf_counter()
    snaps = _load_snapshots(data_dir)
    assert snaps.data_matrix.dtype == pt.float64, "Step 4a requires float64 snapshots"
    train, test = snaps.split(T_SPLIT)
    n_train, n_test = len(train.times), len(test.times)
    print(f"[4a] Train: {n_train} snapshots (t<{T_SPLIT}), last train t={train.times[-1]}  Test: {n_test}")
    timings["load_data_s"] = time.perf_counter() - t0

    train_mean_full = train.data_matrix.mean(dim=1)   # fluctuation reference: FULL-window train mean (phase1 run_v2 convention)
    last_train_snapshot = train.data_matrix[:, -1]

    _before_4, train_post_transient = train.split(4.0)
    windows = {"full": train, "post_transient": train_post_transient}

    dmd_ref = _load_dmd_sweep_reference(str(_PHASE1_DIR / "results_v2" / "sweep.csv"))

    # ---- Grid: window x rank x arm x target ----
    grid_rows: list[dict] = []
    per_window_rank: dict[tuple[str, int], dict] = {}   # for rank-neighbour noise band, arm2 only

    t0 = time.perf_counter()
    for window_name, w in windows.items():
        n_fit = len(w.times)
        pod_flowtorch = fit_pod(w.data_matrix, rank=max(RANKS))   # matches phase1's floor computation exactly

        for rank in RANKS:
            if rank > n_fit - 1:
                continue
            basis = pod_basis(w.data_matrix, rank, center=True)
            Z_fit = project(basis, w.data_matrix)

            pod_floor_pred = pod_projection_floor(pod_flowtorch.modes, pod_flowtorch.mean, test.data_matrix, rank)
            pod_floor = _errors(pod_floor_pred, test.data_matrix, train_mean_full)

            ref_key = (window_name, rank)
            dmd_row = dmd_ref.get(ref_key)

            max_norm_train = Z_fit.norm(dim=0).max().item()
            max_allowed = 1e3 * max_norm_train
            z0 = Z_fit[:, -1]   # last training snapshot (t=7.975), in this window's basis

            for target_kind in TARGETS:
                for arm_name, library in (("arm2", lib2), ("arm3", lib3)):
                    Z_input, Y_full, time_index, trunc = build_target(Z_fit, target_kind, dt=w.dt)
                    D_full, groups = build_regressors(Z_input, library)
                    (D_train, Y_train), (D_val, Y_val) = _train_val_split(D_full, Y_full)

                    lam_lin, lam_quad, lam_diag = select_lambda(
                        Z_fit, library, target_kind, dt=w.dt, grid=LAMBDA_GRID,
                        val_fraction=VAL_FRACTION, rollout_steps=ROLLOUT_STEPS_FOR_SELECTION,
                    )

                    coef_val_model = fit_ridge(D_train, Y_train, lam_lin, lam_quad, groups)
                    train_residual = _rel_l2(coef_val_model @ D_train, Y_train)
                    val_residual = _rel_l2(coef_val_model @ D_val, Y_val)

                    coef_final = fit_ridge(D_full, Y_full, lam_lin, lam_quad, groups)
                    forecast = _forecast_from(coef_final, z0, target_kind, library, basis,
                                               n_test, w.dt, max_allowed, test.data_matrix, train_mean_full)

                    floor_ratio = (forecast["full_field"] / pod_floor["full_field"]
                                   if pod_floor["full_field"] > 0 else float("inf"))

                    row = {
                        "window": window_name, "rank": rank, "arm": arm_name, "target": target_kind,
                        "n_fit": n_fit, "lam_lin": lam_lin, "lam_quad": lam_quad,
                        "train_residual": train_residual, "val_residual": val_residual,
                        "forecast_full_field": forecast["full_field"],
                        "forecast_fluctuation": forecast["fluctuation"],
                        "diverged": forecast["diverged"], "divergence_step": forecast["divergence_step"],
                        "pod_floor_full_field": pod_floor["full_field"],
                        "pod_floor_fluctuation": pod_floor["fluctuation"],
                        "floor_ratio": floor_ratio,
                        "dmd_ref_full_field": dmd_row["primary_full_field"] if dmd_row else None,
                        "dmd_ref_fluctuation": dmd_row["primary_fluctuation"] if dmd_row else None,
                        "truncation_estimate": trunc,
                        "selection_best_score": lam_diag["best_score"],
                    }
                    grid_rows.append(row)

                    if arm_name == "arm2":
                        per_window_rank[(window_name, rank, target_kind)] = {
                            "row": row, "D_full": D_full, "Y_full": Y_full, "groups": groups,
                            "D_train": D_train, "Y_train": Y_train, "D_val": D_val, "Y_val": Y_val,
                            "z0": z0, "basis": basis, "max_allowed": max_allowed, "w": w,
                        }

                    print(f"  [{window_name} r={rank:>2} {arm_name} {target_kind:>10}] "
                          f"lam=({lam_lin:g},{lam_quad:g}) train_res={train_residual:.3e} "
                          f"val_res={val_residual:.3e} forecast(full/fluct)="
                          f"{forecast['full_field']:.3e}/{forecast['fluctuation']:.3e} "
                          f"floor_ratio={floor_ratio:.3f} diverged={forecast['diverged']}")

    timings["grid_s"] = time.perf_counter() - t0

    # ---- Null control: Arm3 regressors/coefs/forecasts == Arm2's on unmodified data ----
    null_control = _run_null_control(windows, lib2, lib3, grid_rows)

    # ---- Sanity checks ----
    t0 = time.perf_counter()
    sanity = _run_sanity_checks(windows, test, train_mean_full, dmd_ref)
    timings["sanity_checks_s"] = time.perf_counter() - t0

    # ---- Noise band (Arm 2 only) ----
    t0 = time.perf_counter()
    noise_band = _run_noise_band(per_window_rank, windows, lib2, test, train_mean_full)
    timings["noise_band_s"] = time.perf_counter() - t0

    # ---- Library ablation (context: which family closes the gap to the floor) ----
    t0 = time.perf_counter()
    ablation = _run_library_ablation(windows, test, train_mean_full)
    timings["ablation_s"] = time.perf_counter() - t0

    timings["total_s"] = time.perf_counter() - t_run_start

    # ---- Write outputs ----
    _write_outputs(results_dir, grid_rows, noise_band, sanity, null_control,
                    case_dir, source_dir, data_dir, timings)
    with open(results_dir / "library_ablation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablation[0].keys()))
        writer.writeheader()
        writer.writerows(ablation)

    return {
        "grid": grid_rows, "noise_band": noise_band, "sanity_checks": sanity,
        "null_control": null_control, "timings": timings,
    }


def _run_null_control(windows, lib2: Library, lib3: Library, grid_rows: list[dict]) -> dict:
    """Arm 3 == Arm 2 by construction on unmodified data. Two parts:

    (1) Regressor-matrix identity: with identical families, build_regressors
    on the SAME Z_input must produce bit-identical D matrices (this is a
    pure, deterministic tensor construction -- no linear solve involved,
    so it is checked exactly, not to a tolerance).

    (2) Coefficient/forecast identity, read off the grid: since arm2 and
    arm3 use the same D/Y/library/lambda-selection for every (window, rank,
    target) cell, their fitted models are the SAME closed-form solve
    applied to the SAME inputs -- but two independently-invoked
    torch.linalg solves on identical inputs are not bit-reproducible under
    multi-threaded BLAS (verified: fitting the same (D, Y) twice differs by
    ~1e-13, and an unregularized lambda=0 fit on the real (noisy) 45-column
    continuous-target regressor problem can diverge in a 20-step RK4
    rollout, which is a numerical-conditioning issue orthogonal to the
    arm2-vs-arm3 question this control checks). So rather than a bespoke
    lambda=0 recompute, this reuses the already-computed, properly
    regularized grid cells (all 16 window x rank x target combinations,
    not just one) and diffs their headline scalar metrics."""
    result = {"families_equal": lib2.families == lib3.families, "regressor_identity": [], "grid_cells": []}

    w = windows["full"]
    rank = RANKS[0]
    basis = pod_basis(w.data_matrix, rank, center=True)
    Z_fit = project(basis, w.data_matrix)
    for target_kind in TARGETS:
        Z_input, _Y, _time_index, _trunc = build_target(Z_fit, target_kind, dt=w.dt)
        D2, _groups2 = build_regressors(Z_input, lib2)
        D3, _groups3 = build_regressors(Z_input, lib3)
        regressors_equal = bool(pt.equal(D2, D3))
        result["regressor_identity"].append({"target": target_kind, "regressors_equal": regressors_equal})
        assert regressors_equal, f"Null control failed: Arm2/Arm3 regressor matrices differ ({target_kind})"

    by_key: dict[tuple, dict] = {}
    for row in grid_rows:
        key = (row["window"], row["rank"], row["target"])
        by_key.setdefault(key, {})[row["arm"]] = row

    tol = 1e-6   # relative tolerance on the headline metrics, well above the ~1e-13 BLAS noise floor
    for key, arms in by_key.items():
        r2, r3 = arms["arm2"], arms["arm3"]
        diffs = {}
        for metric in ("lam_lin", "lam_quad", "train_residual", "val_residual",
                       "forecast_full_field", "forecast_fluctuation"):
            v2, v3 = r2[metric], r3[metric]
            denom = max(abs(v2), abs(v3), 1e-300)
            diffs[metric] = abs(v2 - v3) / denom
        cell = {"window": key[0], "rank": key[1], "target": key[2], "relative_diffs": diffs}
        result["grid_cells"].append(cell)
        for metric, d in diffs.items():
            assert d <= tol, f"Null control failed: {metric} differs by {d:.3e} (>{tol:.0e}) at {key}"

    max_diff = max(max(c["relative_diffs"].values()) for c in result["grid_cells"])
    result["max_relative_diff_over_grid"] = max_diff
    print(f"[4a] Null control PASSED: regressors identical on both targets; "
          f"max relative arm2/arm3 metric diff over the {len(result['grid_cells'])}-cell grid = {max_diff:.2e}")
    return result


def _run_sanity_checks(windows, test, train_mean_full, dmd_ref) -> dict:
    """(a) DMD reproduction: {lin} library, raw (uncentred) basis from the
    SVD of the fit window's X[:, :-1] -- flowTorch's own DMD convention
    (verified by reading flowtorch/analysis/dmd.py: X = data[:, :-1],
    SVD(X, rank), A_tilde = U^T Y V S^-1, which is exactly the lambda=0
    least-squares solution of z_{k+1} = A z_k in that raw basis).
    (b) POD floor: must match phase1/results_v2/sweep.csv pod_floor_* to
    rel 1e-6 (same fit_pod/pod_projection_floor call as phase1's run_v2).
    """
    dmd_rows, floor_rows = [], []
    library_lin = Library(families=("lin",), extras=())

    for window_name, w in windows.items():
        n_fit = len(w.times)
        pod_flowtorch = fit_pod(w.data_matrix, rank=max(RANKS))
        for rank in RANKS:
            if rank > n_fit - 1:
                continue
            ref = dmd_ref.get((window_name, rank))
            if ref is None:
                continue

            # Our lambda=0 {lin} least-squares fit in the raw SVD(X[:, :-1])
            # basis gives flowTorch's reduced operator A_tilde. flowTorch then
            # forecasts through the EXACT DMD modes phi = Y V S^-1 W and
            # b = pinv(phi) x0 (dmd.py _compute_mode_decomposition / predict),
            # so we lift our operator the same way and compare that forecast.
            X_fit, Y_fit = w.data_matrix[:, :-1], w.data_matrix[:, 1:]
            basis_raw = pod_basis(X_fit, rank, center=False)
            Z_raw = project(basis_raw, w.data_matrix)
            D, groups = build_regressors(Z_raw[:, :-1], library_lin)
            coef = fit_ridge(D, Z_raw[:, 1:], 0.0, 0.0, groups)
            A_tilde = coef[:, groups["lin"]]
            _U, S_full, Vh = pt.linalg.svd(X_fit, full_matrices=False)
            V, s_inv = Vh[:rank].T, pt.diag(1.0 / S_full[:rank])
            eigvals, eigvecs = pt.linalg.eig(A_tilde)
            cdt = eigvals.dtype
            phi = Y_fit.to(cdt) @ V.to(cdt) @ s_inv.to(cdt) @ eigvecs
            b = pt.linalg.pinv(phi) @ w.data_matrix[:, -1].to(cdt)
            pred = (phi @ pt.diag(b) @ pt.linalg.vander(eigvals, N=N_TEST + 1)).real[:, 1:]
            diverged = not bool(pt.isfinite(pred).all())
            got_full = relative_l2_error(pred, test.data_matrix)
            got_fluct = fluctuation_relative_error(pred, test.data_matrix, train_mean_full)
            # Projected-basis rollout (x = U z) of the same operator, for the record:
            # it differs from the exact-mode forecast because phi != U W.
            traj, _div, _ = rollout(coef, Z_raw[:, -1], N_TEST, "discrete", library_lin)
            projected_full = relative_l2_error(basis_raw.modes @ traj[:, 1:], test.data_matrix)
            rel_diff_full = abs(got_full - ref["primary_full_field"]) / max(abs(ref["primary_full_field"]), 1e-300)
            rel_diff_fluct = abs(got_fluct - ref["primary_fluctuation"]) / max(abs(ref["primary_fluctuation"]), 1e-300)
            dmd_rows.append({
                "window": window_name, "rank": rank, "diverged": diverged,
                "got_full_field": got_full, "ref_full_field": ref["primary_full_field"],
                "rel_diff_full_field": rel_diff_full,
                "got_fluctuation": got_fluct, "ref_fluctuation": ref["primary_fluctuation"],
                "rel_diff_fluctuation": rel_diff_fluct,
                "projected_rollout_full_field": projected_full,
            })

            pod_floor_pred = pod_projection_floor(pod_flowtorch.modes, pod_flowtorch.mean, test.data_matrix, rank)
            floor_err = _errors(pod_floor_pred, test.data_matrix, train_mean_full)
            floor_rel_full = abs(floor_err["full_field"] - ref["pod_floor_full_field"]) / max(abs(ref["pod_floor_full_field"]), 1e-300)
            floor_rel_fluct = abs(floor_err["fluctuation"] - ref["pod_floor_fluctuation"]) / max(abs(ref["pod_floor_fluctuation"]), 1e-300)
            floor_rows.append({
                "window": window_name, "rank": rank,
                "got_full_field": floor_err["full_field"], "ref_full_field": ref["pod_floor_full_field"],
                "rel_diff_full_field": floor_rel_full,
                "got_fluctuation": floor_err["fluctuation"], "ref_fluctuation": ref["pod_floor_fluctuation"],
                "rel_diff_fluctuation": floor_rel_fluct,
            })

    max_dmd_rel = max((max(r["rel_diff_full_field"], r["rel_diff_fluctuation"]) for r in dmd_rows), default=float("nan"))
    max_floor_rel = max((max(r["rel_diff_full_field"], r["rel_diff_fluctuation"]) for r in floor_rows), default=float("nan"))

    print(f"[4a] Sanity (a) DMD reproduction: max relative mismatch = {max_dmd_rel:.3e} (target <=1e-6)")
    print(f"[4a] Sanity (b) POD floor:        max relative mismatch = {max_floor_rel:.3e} (target <=1e-6)")

    assert max_floor_rel <= 1e-6, (
        f"Sanity check (b) POD floor failed: max relative mismatch {max_floor_rel:.3e} > 1e-6. "
        f"Rows: {floor_rows}"
    )

    assert max_dmd_rel <= 1e-6, (
        f"Sanity check (a) DMD reproduction failed: max relative mismatch {max_dmd_rel:.3e} > 1e-6. "
        f"Rows: {dmd_rows}"
    )

    return {
        "dmd_reproduction": {
            "tolerance": 1e-6, "passed": True, "max_relative_mismatch": max_dmd_rel,
            "rows": dmd_rows,
            "note": (
                "lambda=0 {lin} least squares in the raw SVD(X[:,:-1]) basis equals flowTorch's "
                "A_tilde = U^T Y V S^-1; lifted through the exact DMD modes phi = Y V S^-1 W with "
                "b = pinv(phi) x0 it reproduces phase1/results_v2 primary errors. The projected "
                "rollout x = U z of the same operator (projected_rollout_full_field) differs by up "
                "to ~3% because phi != U W."
            ),
        },
        "pod_floor": {
            "tolerance": 1e-6, "passed": True, "max_relative_mismatch": max_floor_rel, "rows": floor_rows,
        },
    }


def _run_noise_band(per_window_rank, windows, lib2: Library, test, train_mean_full) -> dict:
    """Noise band for Step 5, Arm 2 only, per (window, rank, target) cell:
    leave-one-block-out over 5 contiguous TRAINING blocks (of the 80% split
    used for lambda selection), lambda x10/lambda/10 component-wise
    (lambda=0 -> neighbour 1e-8), and neighbouring ranks in RANKS. Records
    min/max/std of val_residual, forecast_full_field, forecast_fluctuation."""
    band: dict[str, dict] = {}
    rank_index = {r: i for i, r in enumerate(RANKS)}

    for (window_name, rank, target_kind), ctx in per_window_rank.items():
        base_row = ctx["row"]
        groups = ctx["groups"]
        D_train, Y_train = ctx["D_train"], ctx["Y_train"]
        D_val, Y_val = ctx["D_val"], ctx["Y_val"]
        z0, basis, max_allowed, w = ctx["z0"], ctx["basis"], ctx["max_allowed"], ctx["w"]
        lam_lin, lam_quad = base_row["lam_lin"], base_row["lam_quad"]

        def _cell_metrics(coef_val, coef_forecast) -> dict:
            val_res = _rel_l2(coef_val @ D_val, Y_val)
            fc = _forecast_from(coef_forecast, z0, target_kind, lib2, basis, N_TEST, w.dt,
                                 max_allowed, test.data_matrix, train_mean_full)
            return {"val_residual": val_res, "forecast_full_field": fc["full_field"],
                    "forecast_fluctuation": fc["fluctuation"]}

        # --- leave-one-block-out over the training blocks ---
        loo_metrics: dict[str, list[float]] = {"val_residual": [], "forecast_full_field": [], "forecast_fluctuation": []}
        for lo, hi in _contiguous_blocks(D_train.shape[1]):
            keep = list(range(0, lo)) + list(range(hi, D_train.shape[1]))
            if len(keep) < 5:
                continue
            idx = pt.tensor(keep, dtype=pt.long)
            D_loo, Y_loo = D_train.index_select(1, idx), Y_train.index_select(1, idx)
            coef_loo = fit_ridge(D_loo, Y_loo, lam_lin, lam_quad, groups)
            m = _cell_metrics(coef_loo, coef_loo)
            for k, v in m.items():
                loo_metrics[k].append(v)

        # --- lambda x10 / /10, component-wise (0 -> neighbour 1e-8) ---
        def _neighbours(lam: float) -> tuple[float, float]:
            base = 1e-8 if lam == 0.0 else lam
            return base * 10.0, base / 10.0

        lam_lin_hi, lam_lin_lo = _neighbours(lam_lin)
        lam_quad_hi, lam_quad_lo = _neighbours(lam_quad)
        lambda_variants = {
            "lam_lin_x10": (lam_lin_hi, lam_quad), "lam_lin_div10": (lam_lin_lo, lam_quad),
            "lam_quad_x10": (lam_lin, lam_quad_hi), "lam_quad_div10": (lam_lin, lam_quad_lo),
        }
        lambda_metrics: dict[str, list[float]] = {"val_residual": [], "forecast_full_field": [], "forecast_fluctuation": []}
        lambda_detail = {}
        for name, (ll, lq) in lambda_variants.items():
            coef_val = fit_ridge(D_train, Y_train, ll, lq, groups)
            coef_full = fit_ridge(ctx["D_full"], ctx["Y_full"], ll, lq, groups)
            m = _cell_metrics(coef_val, coef_full)
            lambda_detail[name] = {"lam_lin": ll, "lam_quad": lq, **m}
            for k, v in m.items():
                lambda_metrics[k].append(v)

        # --- neighbouring ranks (reuse grid results, arm2 only) ---
        i = rank_index[rank]
        neighbour_ranks = [RANKS[j] for j in (i - 1, i + 1) if 0 <= j < len(RANKS)]
        rank_metrics: dict[str, list[float]] = {"val_residual": [], "forecast_full_field": [], "forecast_fluctuation": []}
        for nr in neighbour_ranks:
            other = per_window_rank.get((window_name, nr, target_kind))
            if other is None:
                continue
            rank_metrics["val_residual"].append(other["row"]["val_residual"])
            rank_metrics["forecast_full_field"].append(other["row"]["forecast_full_field"])
            rank_metrics["forecast_fluctuation"].append(other["row"]["forecast_fluctuation"])

        def _spread(values: list[float]) -> dict:
            finite = [v for v in values if v == v and v not in (float("inf"), float("-inf"))]
            if not finite:
                return {"values": values, "min": None, "max": None, "std": None}
            t = pt.tensor(finite, dtype=pt.float64)
            return {"values": values, "min": t.min().item(), "max": t.max().item(),
                    "std": (t.std(unbiased=False).item() if len(finite) > 1 else 0.0)}

        key = f"{window_name}|{rank}|{target_kind}"
        band[key] = {
            "window": window_name, "rank": rank, "target": target_kind,
            "base_cell": {"val_residual": base_row["val_residual"],
                          "forecast_full_field": base_row["forecast_full_field"],
                          "forecast_fluctuation": base_row["forecast_fluctuation"],
                          "lam_lin": lam_lin, "lam_quad": lam_quad},
            "leave_one_block_out": {k: _spread(v) for k, v in loo_metrics.items()},
            "lambda_neighbors": {"detail": lambda_detail, **{k: _spread(v) for k, v in lambda_metrics.items()}},
            "rank_neighbors": {"ranks": neighbour_ranks, **{k: _spread(v) for k, v in rank_metrics.items()}},
        }
    return band


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

_GRID_FIELDS = [
    "window", "rank", "arm", "target", "n_fit", "lam_lin", "lam_quad",
    "train_residual", "val_residual", "forecast_full_field", "forecast_fluctuation",
    "diverged", "divergence_step", "pod_floor_full_field", "pod_floor_fluctuation",
    "floor_ratio", "dmd_ref_full_field", "dmd_ref_fluctuation", "truncation_estimate",
]


ABLATION_LIBRARIES = (
    ("lin_raw", ("lin",), False),                 # DMD-like operator, uncentred basis
    ("lin", ("lin",), True),
    ("const_lin", ("const", "lin"), True),
    ("const_lin_quad", ("const", "lin", "quad"), True),   # == Arm 2 on this case
)


def _run_library_ablation(windows, test, train_mean_full) -> list[dict]:
    """Discrete target only, same lambda selection as the grid. Not part of
    the Arm 2 vs Arm 3 comparison: it shows which regressor family moves
    the forecast towards the POD floor."""
    rows = []
    for window_name, w in windows.items():
        pod_flowtorch = fit_pod(w.data_matrix, rank=max(RANKS))
        for rank in RANKS:
            floor_pred = pod_projection_floor(pod_flowtorch.modes, pod_flowtorch.mean, test.data_matrix, rank)
            floor = _errors(floor_pred, test.data_matrix, train_mean_full)
            for name, families, center in ABLATION_LIBRARIES:
                library = Library(families=families, extras=())
                basis = pod_basis(w.data_matrix, rank, center=center)
                Z_fit = project(basis, w.data_matrix)
                lam_lin, lam_quad, _ = select_lambda(Z_fit, library, "discrete", dt=w.dt, grid=LAMBDA_GRID,
                                                     val_fraction=VAL_FRACTION,
                                                     rollout_steps=ROLLOUT_STEPS_FOR_SELECTION)
                Z_input, Y, _, _ = build_target(Z_fit, "discrete")
                D, groups = build_regressors(Z_input, library)
                (D_train, Y_train), (D_val, Y_val) = _train_val_split(D, Y)
                val_residual = _rel_l2(fit_ridge(D_train, Y_train, lam_lin, lam_quad, groups) @ D_val, Y_val)
                coef = fit_ridge(D, Y, lam_lin, lam_quad, groups)
                max_allowed = 1e3 * Z_fit.norm(dim=0).max().item()
                fc = _forecast_from(coef, Z_fit[:, -1], "discrete", library, basis, N_TEST, w.dt,
                                    max_allowed, test.data_matrix, train_mean_full)
                rows.append({
                    "window": window_name, "rank": rank, "library": name,
                    "lam_lin": lam_lin, "lam_quad": lam_quad, "val_residual": val_residual,
                    "forecast_full_field": fc["full_field"], "forecast_fluctuation": fc["fluctuation"],
                    "pod_floor_full_field": floor["full_field"],
                    "floor_ratio": fc["full_field"] / floor["full_field"], "diverged": fc["diverged"],
                })
                print(f"  [ablation {window_name} r={rank:>2} {name:>14}] val_res={val_residual:.3e} "
                      f"floor_ratio={rows[-1]['floor_ratio']:.2f}")
    return rows


def _write_outputs(results_dir: Path, grid_rows, noise_band, sanity, null_control,
                    case_dir, source_dir, data_dir, timings) -> None:
    with open(results_dir / "grid.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_GRID_FIELDS)
        writer.writeheader()
        for row in grid_rows:
            writer.writerow({k: row[k] for k in _GRID_FIELDS})

    with open(results_dir / "grid.json", "w") as f:
        json.dump(grid_rows, f, indent=2)

    with open(results_dir / "noise_band.json", "w") as f:
        json.dump(noise_band, f, indent=2)

    with open(results_dir / "sanity_checks.json", "w") as f:
        json.dump({**sanity, "null_control": null_control}, f, indent=2)

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_PHASE2_DIR.parent), text=True
        ).strip()
    except Exception:
        git_hash = "unknown"

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "git_hash": git_hash,
        "torch_version": pt.__version__,
        "case_dir": case_dir, "source_dir": source_dir, "data_dir": data_dir,
        "ranks": list(RANKS), "targets": list(TARGETS),
        "t_split": T_SPLIT, "n_test": N_TEST, "val_fraction": VAL_FRACTION,
        "lambda_grid": list(LAMBDA_GRID), "n_loo_blocks": N_LOO_BLOCKS,
        "rollout_steps_for_selection": ROLLOUT_STEPS_FOR_SELECTION,
        "timings_s": timings,
    }
    try:
        import flowtorch
        metadata["flowtorch_version"] = getattr(flowtorch, "__version__", "unknown")
    except Exception:
        pass
    with open(results_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[4a] Wrote results to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4a: static OpInf null control (Arm2 vs Arm3).")
    parser.add_argument("--case-dir", default=DEFAULT_CASE_DIR, help="OpenFOAM case directory (for case_activity)")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR, help="fixture dir with UEqn.H/pEqn.H/fvSchemes/turbulenceProperties")
    parser.add_argument("--data-dir", default=None, help="explicit dataset case path (default: flowTorch's registered of_cylinder2D_binary)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output folder under phase2/ (default: results_step4a)")
    args = parser.parse_args()

    run_step4a(case_dir=args.case_dir, source_dir=args.source_dir, data_dir=args.data_dir, out_dir=args.out)
