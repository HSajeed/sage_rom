"""
Loads the flowTorch cylinder2D dataset (real OpenFOAM pimpleFoam run, Re=100,
401 snapshots at dt=0.025s -- cell count not independently confirmed here,
check with `loader.vertices.shape[0]` on real data rather than trusting this
docstring) and assembles a snapshot matrix from
the VELOCITY field, U.

Why U and not vorticity: flowTorch's own tutorials build their data matrix
from vorticity because it's a nice single scalar-like quantity for a first
SVD/DMD demo. We deliberately use U instead, because U (and p) are the actual
state variables the Phase 2 operator graph is built around -- fvm::ddt(U),
fvm::div(phi,U), fvm::laplacian(nu,U) all act on U directly. Keeping Phase 1's
baselines and Phase 2's arms operating on the same state variable is what
makes the eventual comparison meaningful; benchmarking POD/DMD on vorticity
while the solver-derived ROM predicts U would be comparing different things.

SETUP (you run this part -- not done in this scaffold):
    pip install flowtorch-fluid   # PyPI distribution name; import name is still `flowtorch`

    Download the flowTorch datasets repository (see flowTorch's README for
    the current location/instructions) and set:
        export FLOWTORCH_DATASETS=/path/to/flowtorch/datasets

    Verify with:
        python -c "from flowtorch import DATASETS; print('of_cylinder2D_binary' in DATASETS)"
    This must print True before anything below will run.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch as pt


@dataclass
class CylinderSnapshots:
    data_matrix: pt.Tensor      # (2 * n_selected_cells, n_snapshots) -- [u_x; u_y] stacked
    times: list[float]           # snapshot times, same order as columns
    n_cells_selected: int
    mask: pt.Tensor               # boolean mask into the full mesh
    vertices: pt.Tensor            # (n_selected_cells, 2) masked cell-center coords, for plotting
    dt: float

    def split(self, t_split: float) -> tuple["CylinderSnapshots", "CylinderSnapshots"]:
        """Time-based train/test split -- train on t < t_split, test (extrapolation) on t >= t_split.
        This is the held-out generalization axis for this dataset: a single
        trajectory, not multiple Reynolds numbers, so "does the ROM generalize"
        here means "does it forecast forward in time it never trained on,"
        not "does it generalize across flow regimes." Keep that distinction in
        mind when writing up results -- it's a real but narrower claim than
        cross-Re generalization would be.
        """
        train_idx = [i for i, t in enumerate(self.times) if t < t_split]
        test_idx = [i for i, t in enumerate(self.times) if t >= t_split]
        train = CylinderSnapshots(
            data_matrix=self.data_matrix[:, train_idx],
            times=[self.times[i] for i in train_idx],
            n_cells_selected=self.n_cells_selected, mask=self.mask,
            vertices=self.vertices, dt=self.dt,
        )
        test = CylinderSnapshots(
            data_matrix=self.data_matrix[:, test_idx],
            times=[self.times[i] for i in test_idx],
            n_cells_selected=self.n_cells_selected, mask=self.mask,
            vertices=self.vertices, dt=self.dt,
        )
        return train, test


def load_cylinder_snapshots(
    t_min: float | None = None,
    domain_lower: tuple[float, float] = (0.1, -1.0),
    domain_upper: tuple[float, float] = (0.75, 1.0),
) -> CylinderSnapshots:
    """
    t_min default (None) selects the first available time step excluding 0,
    so the snapshot window includes the transient rather than starting at the
    developed periodic regime. Pass an explicit t_min (e.g. 4.0, which matches
    flowTorch's own tutorials: vortex shedding starts ~t=1.5s and is fully
    developed by ~t=4s) to restrict to the post-transient periodic regime.
    domain_lower/upper matches the tutorials' spatial crop (1d before, 7.5d
    after the cylinder center) -- change if you want the full domain, but the
    default keeps this consistent with the documented, verified example.
    """
    from flowtorch import DATASETS
    from flowtorch.data import FOAMDataloader, mask_box

    if "of_cylinder2D_binary" not in DATASETS:
        raise RuntimeError(
            "Dataset 'of_cylinder2D_binary' not found in flowtorch.DATASETS. "
            "This means FLOWTORCH_DATASETS isn't set, or doesn't point at a "
            "directory containing that dataset. See this module's docstring."
        )

    path = DATASETS["of_cylinder2D_binary"]
    loader = FOAMDataloader(path)

    all_times = loader.write_times
    if t_min is None:
        # First available time step excluding 0, so the window includes the
        # transient (t_min=4.0 in the tutorials deliberately drops it).
        nonzero = [t for t in all_times if float(t) != 0.0]
        window_times = nonzero
    else:
        window_times = [t for t in all_times if float(t) >= t_min]
    dt = float(window_times[1]) - float(window_times[0])

    vertices = loader.vertices[:, :2]
    mask = mask_box(vertices, lower=list(domain_lower), upper=list(domain_upper))
    n_selected = int(mask.sum().item())

    # U is a 3-component field even in this 2D case (OpenFOAM always stores
    # 3 components; z is ~0 here). Stack [u_x; u_y] into one column per
    # snapshot -- shape (2 * n_selected, n_snapshots).
    data_matrix = pt.zeros((2 * n_selected, len(window_times)), dtype=pt.float32)
    for i, t in enumerate(window_times):
        u = loader.load_snapshot("U", t)          # (n_cells_full, 3)
        u_x = pt.masked_select(u[:, 0], mask)
        u_y = pt.masked_select(u[:, 1], mask)
        data_matrix[:n_selected, i] = u_x
        data_matrix[n_selected:, i] = u_y

    return CylinderSnapshots(
        data_matrix=data_matrix,
        times=[float(t) for t in window_times],
        n_cells_selected=n_selected,
        mask=mask,
        vertices=vertices[mask],  # (n_selected, 2), for plotting
        dt=dt,
    )


if __name__ == "__main__":
    snaps = load_cylinder_snapshots()
    print(f"Data matrix: {tuple(snaps.data_matrix.shape)}  "
          f"({snaps.n_cells_selected} cells x 2 components, {len(snaps.times)} snapshots)")
    print(f"Time range: {snaps.times[0]}s to {snaps.times[-1]}s, dt={snaps.dt}s")
    train, test = snaps.split(t_split=8.0)
    print(f"Train: {train.data_matrix.shape[1]} snapshots (t < 8.0s)")
    print(f"Test:  {test.data_matrix.shape[1]} snapshots (t >= 8.0s, held out for extrapolation)")
