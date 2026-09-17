"""
Shared metrics harness -- the four axes from the original Phase 1 scoping
discussion: reconstruction accuracy, extrapolation (generalization in time,
for this single-trajectory dataset -- see data_loading.py's docstring on
why that's a narrower claim than cross-Reynolds-number generalization),
data efficiency, and wall-clock cost. Deliberately baseline-agnostic: POD,
DMD, and the neural ROM have different fit/forecast signatures (POD has no
forecast at all), so this module works with plain tensors and small
closures rather than forcing a common model interface that would be
awkward for at least one of the three.
"""

from __future__ import annotations

import time

import torch as pt


def relative_l2_error(pred: pt.Tensor, true: pt.Tensor) -> float:
    return (pt.linalg.norm(pred - true) / pt.linalg.norm(true)).item()


def fluctuation_relative_error(pred: pt.Tensor, true: pt.Tensor, reference_mean: pt.Tensor) -> float:
    """Error normalised by the fluctuation of the true data about the
    TRAINING temporal mean, not the test data's own mean -- this is what
    makes the mean predictor score exactly 1.0 (its numerator equals its
    denominator by construction) and gives a meaningful floor: any model
    scoring >=1.0 here is not doing better than predicting the training
    mean, regardless of what its full-field relative_l2_error says."""
    mean_col = reference_mean.unsqueeze(-1) if reference_mean.dim() == 1 else reference_mean
    denom = pt.linalg.norm(true - mean_col)
    return (pt.linalg.norm(pred - true) / denom).item()


def mean_predictor(train_mean: pt.Tensor, n_steps: int) -> pt.Tensor:
    """Repeats the training temporal mean for n_steps columns -- the
    null forecast baseline any real model must beat."""
    return train_mean.unsqueeze(-1).expand(-1, n_steps)


def persistence_predictor(last_snapshot: pt.Tensor, n_steps: int) -> pt.Tensor:
    """Repeats the last training snapshot for n_steps columns -- the
    "nothing changes" baseline."""
    return last_snapshot.unsqueeze(-1).expand(-1, n_steps)


def pod_projection_floor(pod_modes: pt.Tensor, mean: pt.Tensor, test: pt.Tensor, r: int) -> pt.Tensor:
    """POD(train) projection of test data onto the first r modes:
    mean + U_r U_r^T (test - mean). This is the best any rank-r LINEAR
    subspace model fit on the training data could possibly do on the test
    data -- a floor, not a forecast (see pod_baseline.py's module
    docstring on why POD itself has no forecast)."""
    Ur = pod_modes[:, :r]
    centered = test - mean.unsqueeze(-1)
    return mean.unsqueeze(-1) + Ur @ (Ur.T @ centered)


def time_call(fn, *args, **kwargs) -> tuple:
    """Returns (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def data_efficiency_curve(forecast_with_n_snapshots, test_data: pt.Tensor, snapshot_counts: list[int]) -> dict[int, float]:
    """
    forecast_with_n_snapshots: callable(n) -> forecast tensor for the test
    window, using only the first n training snapshots to fit. The caller
    defines this per-baseline (see run_phase1.py) since fitting/forecasting
    differs between POD, DMD, and the neural ROM.
    """
    errors = {}
    for n in snapshot_counts:
        pred = forecast_with_n_snapshots(n)
        errors[n] = relative_l2_error(pred, test_data)
    return errors


def print_summary_table(results: dict) -> None:
    """results: {baseline_name: {metric_name: value}}. Prints a simple
    aligned table; None entries print as 'n/a' (e.g. POD has no
    extrapolation number -- an honest gap, not a missing computation)."""
    metrics = sorted({m for r in results.values() for m in r})
    name_w = max(len(n) for n in results) + 2
    metric_w = 22

    header = "baseline".ljust(name_w) + "".join(m.ljust(metric_w) for m in metrics)
    print(header)
    print("-" * len(header))
    for name, row in results.items():
        line = name.ljust(name_w)
        for m in metrics:
            v = row.get(m)
            cell = "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
            line += cell.ljust(metric_w)
        print(line)
