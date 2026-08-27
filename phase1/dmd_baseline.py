"""
DMD baseline. flowtorch.analysis.DMD exposes eigvals/modes/amplitude/dynamics
and a `.reconstruction` covering only the snapshots it was fit on, plus a
`.predict(initial_condition, n_steps)` method for forecasting forward from a
given state. `forecast()` below wraps `.predict()` directly.

IMPORTANT, found while building this: `.reconstruction`/`.dynamics` are NOT
reliable in this version of flowtorch -- verified with a from-scratch exact
linear system (x_{n+1} = A @ x_n, no noise) where `.predict()` recovered the
true trajectory to 1e-6 relative error, while `.reconstruction` on the exact
same fit was off by 139%. Root cause traced into the library source
(dmd.py): `dynamics`/`reconstruction` build their Vandermonde matrix with
the legacy `torch.vander` (descending powers by default), while `.predict()`
uses `torch.linalg.vander` (increasing powers) -- two incompatible
conventions inside the same class. Use `.predict()` (i.e. this module's
`forecast()`) for anything that needs to be numerically trustworthy;
`.reconstruction`/`.dynamics` are fine for the qualitative mode-shape plots
flowTorch's own tutorials use them for, but do not use them as a ground-
truth reference for validating anything else, including this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch as pt
from flowtorch.analysis import DMD


@dataclass
class DMDModel:
    dmd: DMD
    t0: float    # time of the first training snapshot -- forecast() steps are relative to this
    dt: float


def fit_dmd(data_matrix: pt.Tensor, times: list[float], rank: int) -> DMDModel:
    dt = times[1] - times[0]
    dmd = DMD(data_matrix, dt=dt, rank=rank)
    return DMDModel(dmd=dmd, t0=times[0], dt=dt)


def forecast(model: DMDModel, initial_condition: pt.Tensor, n_steps: int) -> pt.Tensor:
    """Forecast n_steps+1 states (including initial_condition as step 0),
    spaced by the fitted dt. Thin wrapper around DMD.predict() -- see
    module docstring for why this delegates rather than reimplementing the
    amplitude/Vandermonde math."""
    return model.dmd.predict(initial_condition, n_steps)


if __name__ == "__main__":
    from test_synthetic import make_synthetic_snapshots
    from pod_baseline import select_rank_by_energy

    snaps = make_synthetic_snapshots()
    train, test = snaps.split(t_split=snaps.times[int(0.7 * len(snaps.times))])

    rank = select_rank_by_energy(train.data_matrix, 0.99)
    model = fit_dmd(train.data_matrix, train.times, rank=rank)
    print(f"DMD fit: rank={rank}, {len(train.times)} training snapshots")

    n_train, n_test = len(train.times), len(test.times)
    full_forecast = forecast(model, train.data_matrix[:, 0], n_train + n_test - 1)

    # Check against the actual known data, not against dmd.reconstruction
    # (see module docstring for why that would be a misleading check).
    train_err = (pt.linalg.norm(full_forecast[:, :n_train] - train.data_matrix)
                 / pt.linalg.norm(train.data_matrix)).item()
    test_err = (pt.linalg.norm(full_forecast[:, n_train:] - test.data_matrix)
                / pt.linalg.norm(test.data_matrix)).item()
    print(f"Fit error on training window (forecast vs. true data):    {train_err:.4f}")
    print(f"Extrapolation error on held-out window (forecast vs. true data): {test_err:.4f}")
