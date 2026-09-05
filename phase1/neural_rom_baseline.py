"""
Black-box neural ROM: linear encoder/decoder + a single unstructured MLP for
the latent dynamics dz/dt. Deliberately NOT built from composable operator
blocks (contrast with phase2/rom/composable_operator_rom.py) -- this is
Phase 2's Arm 1, the no-structure-at-all baseline both the hand-specified
and solver-derived arms are meant to beat. Giving this baseline any
hand-imposed structure would undercut the whole comparison.

Uses the same explicit-Euler rollout convention as
composable_operator_rom.py's ComposableOperatorROM.rollout so results are
directly comparable once real training data is available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch as pt
import torch.nn as nn


class NeuralROM(nn.Module):
    def __init__(self, n_features: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Linear(n_features, latent_dim)
        self.decoder = nn.Linear(latent_dim, n_features)
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def encode(self, x: pt.Tensor) -> pt.Tensor:
        return self.encoder(x)

    def decode(self, z: pt.Tensor) -> pt.Tensor:
        return self.decoder(z)

    def rollout_latent(self, z0: pt.Tensor, n_steps: int, dt: float) -> pt.Tensor:
        traj = [z0]
        z = z0
        for _ in range(n_steps):
            z = z + dt * self.dynamics(z)
            traj.append(z)
        return pt.stack(traj, dim=0)   # (n_steps+1, latent_dim)


@dataclass
class NeuralROMModel:
    net: NeuralROM
    dt: float
    t0: float
    latent_dim: int


def _resolve_device(device: str | pt.device | None) -> pt.device:
    if device is not None:
        return pt.device(device)
    return pt.device("cuda" if pt.cuda.is_available() else "cpu")


def fit_neural_rom(
    data_matrix: pt.Tensor, times: list[float], latent_dim: int,
    epochs: int = 500, lr: float = 1e-3, rollout_horizon: int = 10,
    verbose: bool = False,
) -> tuple[NeuralROMModel, dict]:
    """
    data_matrix: (n_features, n_snapshots), columns in time order, evenly
    spaced by dt = times[1]-times[0] (matches this dataset's fixed dt=0.025s).
    Trains on a combination of single-snapshot reconstruction loss and
    short-rollout prediction loss, both against the training window only.
    Uses CUDA automatically when available (falls back to CPU otherwise).

    IMPORTANT, found while testing this against a synthetic fixture: low
    training loss here does NOT imply accurate full-length rollout. The
    rollout loss only ever supervises `rollout_horizon` steps at a time
    (default 10) from a handful of random starting points each epoch --
    never the full training trajectory end to end. On a 210-snapshot
    synthetic test, training loss reached 0.055 (from 1.68) but a full
    rollout across all 210 training steps still had ~27% relative error,
    because rollout error compounds over horizons the loss never directly
    penalized. This is a real property of this training regime, not a bug
    (verified: one-step-ahead error from TRUE encoded states, i.e. no
    compounding, was in the same ballpark -- ~30% -- confirming it's a
    genuine local-dynamics accuracy limit at this epoch count/capacity, not
    drift specifically). If you need accurate long-horizon forecasts,
    increase `rollout_horizon` towards your actual forecast length (more
    expensive per epoch) rather than assuming more epochs alone will fix it.
    """
    n_features, n_snapshots = data_matrix.shape
    dt = times[1] - times[0]
    device = _resolve_device(None)
    x = data_matrix.T.to(device)   # (n_snapshots, n_features), one row per snapshot

    net = NeuralROM(n_features, latent_dim).to(device)
    opt = pt.optim.Adam(net.parameters(), lr=lr)

    history = {"loss": []}
    for epoch in range(epochs):
        opt.zero_grad()

        z = net.encode(x)                          # (n_snapshots, latent_dim)
        recon = net.decode(z)
        recon_loss = ((recon - x) ** 2).mean()

        # short-rollout loss: pick random start indices, roll latent dynamics
        # forward, compare decoded rollout to the true snapshots
        max_start = n_snapshots - rollout_horizon - 1
        if max_start > 0:
            starts = pt.randint(0, max_start, (min(8, max_start),), device=device)
            rollout_loss = 0.0
            for s in starts:
                z0 = net.encode(x[s:s+1])
                z_traj = net.rollout_latent(z0.squeeze(0), rollout_horizon, dt)
                x_pred = net.decode(z_traj)
                x_true = x[s:s+rollout_horizon+1]
                rollout_loss = rollout_loss + ((x_pred - x_true) ** 2).mean()
            rollout_loss = rollout_loss / len(starts)
        else:
            rollout_loss = pt.tensor(0.0, device=device)

        loss = recon_loss + rollout_loss
        loss.backward()
        opt.step()
        history["loss"].append(loss.item())
        if verbose and epoch % max(1, epochs // 10) == 0:
            rollout_val = rollout_loss.item() if isinstance(rollout_loss, pt.Tensor) else rollout_loss
            print(f"  epoch {epoch:4d}  recon={recon_loss.item():.5f}  rollout={rollout_val:.5f}")

    return NeuralROMModel(net=net, dt=dt, t0=times[0], latent_dim=latent_dim), history


def forecast(model: NeuralROMModel, last_known_state: pt.Tensor, query_times: list[float]) -> pt.Tensor:
    """Encode the last known (training) state, roll the latent dynamics
    forward to cover query_times, decode back to full state. query_times
    must be evenly spaced by model.dt and start after the encoded state's
    time -- this mirrors how you'd actually deploy a trained ROM (forecast
    forward from the last observation), not how DMD's forecast() works
    (which can evaluate at any time via the closed-form spectral formula)."""
    net = model.net
    device = next(net.parameters()).device
    last_state = last_known_state.to(device)
    with pt.no_grad():
        z0 = net.encode(last_state.unsqueeze(0)).squeeze(0)
        n_steps = len(query_times)
        z_traj = net.rollout_latent(z0, n_steps - 1, model.dt)
        x_pred = net.decode(z_traj)   # (n_steps, n_features)
    return x_pred.T.cpu()   # (n_features, n_steps), matching data_matrix convention


if __name__ == "__main__":
    from test_synthetic import make_synthetic_snapshots

    snaps = make_synthetic_snapshots()
    train, test = snaps.split(t_split=snaps.times[int(0.7 * len(snaps.times))])

    t_start = time.time()
    model, history = fit_neural_rom(train.data_matrix, train.times, latent_dim=6, epochs=300, verbose=True)
    print(f"Training time: {time.time() - t_start:.1f}s, final loss={history['loss'][-1]:.5f}")

    last_state = train.data_matrix[:, -1]
    pred_test = forecast(model, last_state, test.times)
    err = pt.linalg.norm(pred_test - test.data_matrix) / pt.linalg.norm(test.data_matrix)
    print(f"Extrapolation relative L2 error on held-out window: {err.item():.4f}")
