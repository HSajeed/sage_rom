"""
Plain-script (no pytest) checks of the corrected Phase 1 evaluation protocol
(PATH_FORWARD.md Step 1), run on the synthetic fixture only -- never touches
the real dataset. Run with: python test_eval_protocol.py
"""

from __future__ import annotations

import torch as pt

from test_synthetic import make_synthetic_snapshots
from pod_baseline import fit_pod, select_rank_by_energy
from dmd_baseline import fit_dmd, forecast as dmd_forecast
from metrics import fluctuation_relative_error, mean_predictor, pod_projection_floor, relative_l2_error


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    assert condition, f"{name} failed: {detail}"


def main() -> None:
    snaps = make_synthetic_snapshots()
    train, test = snaps.split(t_split=snaps.times[int(0.7 * len(snaps.times))])
    n_test = len(test.times)
    train_mean = train.data_matrix.mean(dim=1)

    # --- s**2 energy rank <= s-based (cumsum(s)) rank, at the same threshold ---
    pod = fit_pod(train.data_matrix)
    s = pod.singular_values
    s_rank = int((pt.cumsum(s, dim=0) / s.sum() < 0.99).sum().item()) + 1
    s2_rank = select_rank_by_energy(train.data_matrix, 0.99)
    check("s**2 energy rank <= s-based energy rank", s2_rank <= s_rank,
          f"s2_rank={s2_rank} s_rank={s_rank}")

    # --- restart forecast from last train snapshot has low error on this
    # near-linear fixture (DMD is exact-ish on it, see test_synthetic.py).
    # Uses a generous fixed rank (10), not s2_rank: s2_rank (4) happens to
    # truncate mid-way through this fixture's second cos/sin conjugate pair
    # (99.98% energy is already reached at rank 4, one snapshot before the
    # pair completes at rank 5), which is exactly the kind of rank-
    # truncation pathology this protocol needs to be robust to reporting,
    # not something this particular check is about. ---
    restart_rank = min(10, train.data_matrix.shape[1] - 1)
    model = fit_dmd(train.data_matrix, train.times, rank=restart_rank)
    pred = dmd_forecast(model, train.data_matrix[:, -1], n_test)
    pred = pred.real if pt.is_complex(pred) else pred
    err = relative_l2_error(pred[:, 1:], test.data_matrix)
    check("restart-from-last-train-snapshot forecast error is low", err < 0.1,
          f"err={err:.4f}")

    # --- fluctuation error of the mean predictor is 1.0 within tolerance ---
    mean_pred = mean_predictor(train_mean, n_test)
    mean_fluct_err = fluctuation_relative_error(mean_pred, test.data_matrix, train_mean)
    check("mean predictor fluctuation error == 1.0", abs(mean_fluct_err - 1.0) < 1e-5,
          f"got {mean_fluct_err}")

    # --- POD projection floor is monotone non-increasing in rank ---
    ranks = [1, 2, 4, 8, s2_rank]
    ranks = sorted(set(r for r in ranks if r <= train.data_matrix.shape[1] - 1))
    pod_full = fit_pod(train.data_matrix, rank=max(ranks))
    floor_errs = [
        relative_l2_error(pod_projection_floor(pod_full.modes, pod_full.mean, test.data_matrix, r), test.data_matrix)
        for r in ranks
    ]
    non_increasing = all(floor_errs[i] >= floor_errs[i + 1] - 1e-9 for i in range(len(floor_errs) - 1))
    check("POD projection floor is monotone non-increasing in rank", non_increasing,
          f"ranks={ranks} errs={[f'{e:.4f}' for e in floor_errs]}")

    print("\nAll test_eval_protocol.py checks passed.")


if __name__ == "__main__":
    main()
