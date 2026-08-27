"""
Synthetic fixture shaped exactly like data_loading.CylinderSnapshots, but
built from a closed-form oscillating field with KNOWN frequencies -- not
just "some array of the right shape." This lets the baselines be checked for
actual correctness (e.g. "does DMD recover a frequency close to the one we
put in") rather than only "does it run without crashing," while never
touching the real dataset or claiming to say anything about real cylinder-
wake physics. Do not mistake results from this fixture for CFD validation.
"""

from __future__ import annotations

import torch as pt

from data_loading import CylinderSnapshots


def make_synthetic_snapshots(
    n_points: int = 60, n_snapshots: int = 300, dt: float = 0.025,
    freqs_hz: tuple[float, float] = (6.0, 12.0), noise_std: float = 0.01,
    seed: int = 0,
) -> CylinderSnapshots:
    """
    Each oscillating component is built as a cos/sin PAIR with linearly
    independent spatial patterns (e.g. sin(x)*cos(wt) + cos(x)*sin(wt)),
    not a bare single-term product. This matters: a lone
    spatial_pattern(x)*cos(wt) term, with no companion sin(wt) term carrying
    an independent spatial pattern, is NOT the trajectory of any autonomous
    linear system -- the instantaneous state doesn't contain enough
    information to determine its own next step, so no linear operator (DMD
    or otherwise) can fit it. An earlier version of this fixture made
    exactly that mistake and DMD correctly failed to fit it (recovered
    eigenvalues had |lambda| as low as 0.06 and wrong frequencies) -- not a
    flowtorch bug or a forecast() bug, a fixture that wasn't actually
    DMD-representable. Verified correct via a from-scratch exact-linear
    ground-truth test (see dmd_baseline.py) before concluding this.
    """
    g = pt.Generator().manual_seed(seed)
    x = pt.linspace(0, 2 * pt.pi, n_points)
    times = [round(i * dt, 6) for i in range(n_snapshots)]
    t = pt.tensor(times)

    def cos_sin_pair(spatial_cos, spatial_sin, freq_hz):
        w = 2 * pt.pi * freq_hz
        return (spatial_cos.unsqueeze(1) * pt.cos(w * t).unsqueeze(0)
                + spatial_sin.unsqueeze(1) * pt.sin(w * t).unsqueeze(0))

    mode1 = cos_sin_pair(pt.sin(x), pt.cos(x), freqs_hz[0])
    mode2 = 0.4 * cos_sin_pair(pt.cos(2 * x), pt.sin(2 * x), freqs_hz[1])
    steady = pt.cos(x / 2).unsqueeze(1).expand(-1, n_snapshots)   # true DC term, fine as-is (real eigenvalue=1)

    u_x = steady + mode1 + mode2 + noise_std * pt.randn(n_points, n_snapshots, generator=g)
    u_y = 0.5 * steady + 0.7 * mode1 - 0.3 * mode2 + noise_std * pt.randn(n_points, n_snapshots, generator=g)

    data_matrix = pt.cat([u_x, u_y], dim=0)   # (2*n_points, n_snapshots)

    return CylinderSnapshots(
        data_matrix=data_matrix.float(),
        times=times,
        n_cells_selected=n_points,
        mask=pt.ones(n_points, dtype=pt.bool),
        vertices=pt.stack([x, pt.zeros_like(x)], dim=1),
        dt=dt,
    )


if __name__ == "__main__":
    snaps = make_synthetic_snapshots()
    print(f"Synthetic fixture: {tuple(snaps.data_matrix.shape)}, "
          f"{len(snaps.times)} snapshots, dt={snaps.dt}, "
          f"known frequencies=(6.0, 12.0) Hz")
