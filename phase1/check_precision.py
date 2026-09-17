"""
Standalone precision diagnostic -- does the phase1/results_v2 sweep's
non-monotone post-transient POD floor (r23 < r32 < r63, impossible for a
nested orthogonal projection) and the high-rank DMD blowups trace back to
flowTorch's economy SVD ("Gram matrix and eigendecomposition") losing
orthogonality on float32 data, or are they real?

Does NOT touch run_phase1.py or results_v2/ (besides writing this script's
own precision_check.json into results_v2/). Loads the real dataset --
requires FLOWTORCH_DATASETS set up (see data_loading.py).

Run: python check_precision.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch as pt
import flowtorch
from flowtorch.analysis import SVD, DMD

from data_loading import load_cylinder_snapshots
from pod_baseline import fit_pod
from dmd_baseline import fit_dmd, forecast as dmd_forecast, eigen_summary
from metrics import relative_l2_error, fluctuation_relative_error

RANKS = (11, 15, 23, 32, 63)
T_SPLIT = 8.0


def orthogonality_loss(U: pt.Tensor) -> float:
    r = U.shape[1]
    gram = U.T @ U
    return pt.linalg.norm(gram - pt.eye(r, dtype=gram.dtype)).item()


def pod_floor_errors(modes: pt.Tensor, mean: pt.Tensor, test: pt.Tensor,
                      train_mean_for_fluct: pt.Tensor, ranks: tuple) -> dict:
    out = {}
    for r in ranks:
        if r > modes.shape[1]:
            continue
        Ur = modes[:, :r]
        pred = mean.unsqueeze(-1) + Ur @ (Ur.T @ (test - mean.unsqueeze(-1)))
        out[str(r)] = {
            "full_field": relative_l2_error(pred, test),
            "fluctuation": fluctuation_relative_error(pred, test, train_mean_for_fluct),
        }
    return out


def reference_svd_f64(data_matrix_f64: pt.Tensor):
    """torch.linalg.svd directly on the centred float64 matrix -- the most
    accurate reference available, bypassing flowTorch's SVD class (and
    hence its Gram-matrix path) entirely."""
    mean = data_matrix_f64.mean(dim=1)
    centered = data_matrix_f64 - mean.unsqueeze(-1)
    U, S, _ = pt.linalg.svd(centered, full_matrices=False)
    return U, S, mean


def analyze_window(name: str, w32, w64, test32, test64,
                    train_mean_f32: pt.Tensor, train_mean_f64: pt.Tensor) -> dict:
    """w32/test32 and w64/test64 come from two SEPARATE loads of the real
    dataset (load_cylinder_snapshots(dtype=pt.float32) and
    dtype=pt.float64) -- not one loaded as float64 and then .double()'d
    from a float32 array, and not one loaded as float64 and .float()'d
    down. This matters: load_cylinder_snapshots's own default is float64
    (see data_loading.py), so deriving "float32" via anything other than
    an explicit dtype=pt.float32 load would silently compare float64
    against itself."""
    n_fit = len(w32.times)
    ranks = tuple(r for r in RANKS if r <= n_fit - 1)
    result = {"n_fit": n_fit, "ranks": list(ranks)}

    # ---- (a) float32, as run_phase1.py used to do it before the float64
    # default ----
    pod32 = fit_pod(w32.data_matrix, rank=max(ranks))
    ortho32 = {str(r): orthogonality_loss(pod32.modes[:, :r]) for r in ranks}
    floor32 = pod_floor_errors(pod32.modes, pod32.mean, test32.data_matrix, train_mean_f32, ranks)

    # ---- (b) float64, still via flowTorch's SVD (Gram-matrix path if it
    # picks 'evd'/'auto') ----
    w_data_f64 = w64.data_matrix
    test_data_f64 = test64.data_matrix
    pod64 = fit_pod(w_data_f64, rank=max(ranks))
    ortho64 = {str(r): orthogonality_loss(pod64.modes[:, :r]) for r in ranks}
    floor64 = pod_floor_errors(pod64.modes, pod64.mean, test_data_f64, train_mean_f64, ranks)

    # ---- (c) torch.linalg.svd directly on centred float64 data (reference,
    # bypasses flowTorch's SVD class / Gram-matrix path entirely) ----
    Uref, Sref, mean_ref = reference_svd_f64(w_data_f64)
    ortho_ref = {str(r): orthogonality_loss(Uref[:, :r]) for r in ranks}
    floor_ref = pod_floor_errors(Uref, mean_ref, test_data_f64, train_mean_f64, ranks)
    s_ratio_ref = {str(r): (Sref[r - 1] / Sref[0]).item() for r in ranks}

    result["orthogonality_loss"] = {"float32_flowtorch_svd": ortho32, "float64_flowtorch_svd": ortho64,
                                     "float64_torch_linalg_svd_reference": ortho_ref}
    result["pod_projection_floor"] = {"float32_flowtorch_svd": floor32, "float64_flowtorch_svd": floor64,
                                       "float64_torch_linalg_svd_reference": floor_ref}
    result["singular_value_ratio_s_r_over_s1_float64_reference"] = s_ratio_ref

    def monotone_nonincreasing(floor: dict) -> bool:
        errs = [floor[str(r)]["full_field"] for r in ranks]
        return all(errs[i] >= errs[i + 1] - 1e-12 for i in range(len(errs) - 1))

    result["pod_floor_monotone"] = {
        "float32_flowtorch_svd": monotone_nonincreasing(floor32),
        "float64_flowtorch_svd": monotone_nonincreasing(floor64),
        "float64_torch_linalg_svd_reference": monotone_nonincreasing(floor_ref),
    }

    # ---- DMD primary protocol at these ranks, float32 vs float64 ----
    last_train_32 = w32.data_matrix[:, -1]
    last_train_64 = w_data_f64[:, -1]
    n_test = len(test32.times)

    def dmd_run(data_matrix, times, last_train, test_data, ref_mean, ranks):
        rows = {}
        for r in ranks:
            model = fit_dmd(data_matrix, times, rank=r)
            eig = eigen_summary(model)
            max_lambda = max((row["magnitude"] for row in eig["eigenvalues"]), default=float("nan"))
            raw = dmd_forecast(model, last_train, n_test)
            pred = raw.real if pt.is_complex(raw) else raw
            pred = pred[:, 1:]
            rows[str(r)] = {
                "full_field": relative_l2_error(pred, test_data),
                "fluctuation": fluctuation_relative_error(pred, test_data, ref_mean),
                "n_unstable": eig["n_unstable"],
                "max_lambda": max_lambda,
            }
        return rows

    dmd32 = dmd_run(w32.data_matrix, w32.times, last_train_32, test32.data_matrix, train_mean_f32, ranks)

    dmd64_status = "ok"
    try:
        dmd64 = dmd_run(w_data_f64, w64.times, last_train_64, test_data_f64, train_mean_f64, ranks)
    except Exception as exc:
        dmd64_status = f"float64 DMD failed directly: {exc!r} -- falling back to fit in float64, forecast/eval in float32"
        dmd64 = {}
        for r in ranks:
            model64 = fit_dmd(w_data_f64, w64.times, rank=r)
            eig = eigen_summary(model64)
            max_lambda = max((row["magnitude"] for row in eig["eigenvalues"]), default=float("nan"))
            raw = dmd_forecast(model64, last_train_32, n_test)
            pred = (raw.real if pt.is_complex(raw) else raw).float()[:, 1:]
            dmd64[str(r)] = {
                "full_field": relative_l2_error(pred, test32.data_matrix),
                "fluctuation": fluctuation_relative_error(pred, test32.data_matrix, train_mean_f32),
                "n_unstable": eig["n_unstable"],
                "max_lambda": max_lambda,
            }

    result["dmd_primary"] = {"float32": dmd32, "float64": dmd64, "float64_status": dmd64_status}

    return result


def main() -> None:
    print(f"flowtorch version: {flowtorch.__version__ if hasattr(flowtorch, '__version__') else 'unknown'}")

    # Two INDEPENDENT loads at explicit dtypes -- see analyze_window's
    # docstring for why this can't be derived from one load + a cast.
    snaps32 = load_cylinder_snapshots(dtype=pt.float32)
    snaps64 = load_cylinder_snapshots(dtype=pt.float64)

    train32, test32 = snaps32.split(T_SPLIT)
    train64, test64 = snaps64.split(T_SPLIT)
    _before_4_32, train_post_transient_32 = train32.split(4.0)
    _before_4_64, train_post_transient_64 = train64.split(4.0)

    windows32 = {"full": train32, "post_transient": train_post_transient_32}
    windows64 = {"full": train64, "post_transient": train_post_transient_64}

    train_mean_f32 = train32.data_matrix.mean(dim=1)
    train_mean_f64 = train64.data_matrix.mean(dim=1)

    results = {}
    for name in windows32:
        w32, w64 = windows32[name], windows64[name]
        print(f"\n=== window={name} n_fit={len(w32.times)} ===")
        results[name] = analyze_window(name, w32, w64, test32, test64, train_mean_f32, train_mean_f64)
        r = results[name]
        for rk in r["ranks"]:
            s = str(rk)
            o32 = r["orthogonality_loss"]["float32_flowtorch_svd"][s]
            o64 = r["orthogonality_loss"]["float64_flowtorch_svd"][s]
            oref = r["orthogonality_loss"]["float64_torch_linalg_svd_reference"][s]
            f32 = r["pod_projection_floor"]["float32_flowtorch_svd"][s]["full_field"]
            f64 = r["pod_projection_floor"]["float64_flowtorch_svd"][s]["full_field"]
            fref = r["pod_projection_floor"]["float64_torch_linalg_svd_reference"][s]["full_field"]
            d32 = r["dmd_primary"]["float32"][s]
            d64 = r["dmd_primary"]["float64"].get(s, {})
            print(f"  rank={rk:>2}  ortho(f32/f64/ref)={o32:.2e}/{o64:.2e}/{oref:.2e}  "
                  f"floor(f32/f64/ref)={f32:.2e}/{f64:.2e}/{fref:.2e}  "
                  f"dmd_full(f32/f64)={d32['full_field']:.2e}/{d64.get('full_field', float('nan')):.2e}  "
                  f"n_unst(f32/f64)={d32['n_unstable']}/{d64.get('n_unstable', 'n/a')}  "
                  f"max|lam|(f32/f64)={d32['max_lambda']:.4f}/{d64.get('max_lambda', float('nan')):.4f}")
        print(f"  POD floor monotone (f32/f64/ref): "
              f"{r['pod_floor_monotone']['float32_flowtorch_svd']}/"
              f"{r['pod_floor_monotone']['float64_flowtorch_svd']}/"
              f"{r['pod_floor_monotone']['float64_torch_linalg_svd_reference']}")

    out = {
        "description": (
            "Precision diagnostic: orthogonality loss of POD modes, POD "
            "projection floor monotonicity, and DMD primary-protocol errors "
            "at fixed ranks, compared across float32 (as run_phase1.py "
            "uses), float64 via flowTorch's SVD class, and a float64 "
            "torch.linalg.svd reference bypassing flowTorch's SVD entirely."
        ),
        "flowtorch_version": flowtorch.__version__ if hasattr(flowtorch, "__version__") else "unknown",
        "t_split": T_SPLIT,
        "ranks_requested": list(RANKS),
        "windows": results,
    }
    out_path = Path(__file__).parent / "results_v2" / "precision_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
