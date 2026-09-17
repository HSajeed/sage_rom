"""
POD baseline. Scope note: POD alone is a static spatial basis, not a
dynamical model -- it has no notion of "forecast." Its role in this
comparison is reconstruction quality vs. rank (how much of the flow's
variance is captured by r modes) and as the rank-selection tool DMD also
uses. Forecasting is DMD's and the neural ROM's job, not POD's -- resist
the temptation to bolt a forecast onto POD just to fill in a metrics table
cell; an honest "n/a, static basis" is more useful than a number that
doesn't mean what the table implies it means.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch as pt
from flowtorch.analysis import SVD


@dataclass
class PODResult:
    svd: SVD
    mean: pt.Tensor            # temporal mean, subtracted before SVD; add back for reconstruction
    rank: int

    @property
    def modes(self) -> pt.Tensor:
        return self.svd.U   # (n_features, rank)

    @property
    def singular_values(self) -> pt.Tensor:
        return self.svd.s

    def cumulative_energy(self) -> pt.Tensor:
        """Cumulative energy captured by the first k modes. Energy is
        variance, i.e. s**2 (the singular values are proportional to
        sqrt(variance) along each mode), not the singular values
        themselves -- using cumsum(s)/sum(s) instead of cumsum(s**2)/sum(s**2)
        silently redefines "99% energy" as something closer to "99% of a
        linear-in-s budget," which picks a far larger rank than intended
        (verified on real data: rank 63 by cumsum(s) vs. rank 11 by
        cumsum(s**2) for the same 99% threshold)."""
        s2 = self.svd.s ** 2
        return pt.cumsum(s2, dim=0) / s2.sum()

    def reconstruct(self, r: int, coeffs: pt.Tensor | None = None) -> pt.Tensor:
        """Reconstruct using the first r modes. If coeffs is None, uses the
        training data's own projection (svd.V * s) -- i.e. reconstructs the
        data POD was fit on. Pass explicit coeffs to project/reconstruct new
        snapshots via `project`."""
        Ur = self.modes[:, :r]
        if coeffs is None:
            coeffs = (self.svd.V[:, :r] * self.svd.s[:r]).T   # (r, n_snapshots)
        return Ur @ coeffs + self.mean.unsqueeze(-1)

    def project(self, r: int, snapshot: pt.Tensor) -> pt.Tensor:
        """Project a (possibly new) snapshot onto the first r modes."""
        Ur = self.modes[:, :r]
        return Ur.T @ (snapshot - self.mean)


def fit_pod(data_matrix: pt.Tensor, rank: int | None = None) -> PODResult:
    mean = data_matrix.mean(dim=1)
    centered = data_matrix - mean.unsqueeze(-1)
    rank = rank or centered.shape[1]
    svd = SVD(centered, rank=rank)
    return PODResult(svd=svd, mean=mean, rank=rank)


def select_rank_by_energy(data_matrix: pt.Tensor, energy_threshold: float = 0.99) -> int:
    """Shared rank-selection utility -- DMD baseline reuses this so POD and
    DMD are truncated to a comparable basis size rather than each picking an
    independently-tuned rank, which would confound "DMD's dynamics help" with
    "DMD happened to get a better basis." """
    pod = fit_pod(data_matrix)
    cum = pod.cumulative_energy()
    r = int((cum < energy_threshold).sum().item()) + 1
    return min(r, pod.rank)


def reconstruction_error_vs_rank(data_matrix: pt.Tensor, ranks: list[int]) -> dict[int, float]:
    pod = fit_pod(data_matrix, rank=max(ranks))
    errors = {}
    for r in ranks:
        recon = pod.reconstruct(r)
        errors[r] = (pt.linalg.norm(data_matrix - recon) / pt.linalg.norm(data_matrix)).item()
    return errors


if __name__ == "__main__":
    # Smoke test on synthetic data -- see test_synthetic.py for the shared fixture.
    from test_synthetic import make_synthetic_snapshots
    snaps = make_synthetic_snapshots()
    r = select_rank_by_energy(snaps.data_matrix, 0.99)
    print(f"Selected rank for 99% energy: {r}")
    errs = reconstruction_error_vs_rank(snaps.data_matrix, [1, 2, 4, r])
    for rank, err in errs.items():
        print(f"  rank={rank:<3} relative reconstruction error={err:.4f}")
