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
