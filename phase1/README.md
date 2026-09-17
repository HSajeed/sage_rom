# SAGE-CFD -- Phase 1: conventional ROM baselines (POD + DMD)

POD and DMD, evaluated on the same task: fit on an early time window of the
cylinder-wake trajectory, forecast the held-out later window, restarting
from the last training snapshot. This is the linear reference Phase 2's
Arm 2 (hand-specified) and Arm 3 (solver-derived) need to beat -- see
`phase1_report.md` for the current, corrected results and
`../PATH_FORWARD.md` for the decision record (this file is a local,
gitignored file -- not on GitHub).

**Scope on this branch (`DMD_POD`):** NeuralROM (black-box latent-dynamics
Arm 1) has been dropped -- see `../PATH_FORWARD.md` §0.1. Its code is
retained in `neural_rom_baseline.py` and is only reachable through the
legacy path (`run_phase1.py --legacy --with-neural`); it is not part of the
current protocol or report.

## Setup

```
pip install flowtorch-fluid
```

**Do not** `pip install flowtorch` (no suffix) from PyPI -- that installs
an unrelated Meta package (normalizing flows for probabilistic
programming), not this one. Confirmed the hard way while building this:
same package name, zero overlap in functionality, no CFD code in it at
all. The real library's PyPI distribution is named `flowtorch-fluid`
(import name is still `flowtorch`) -- confirmed by installing it directly
and checking it exposes the same `FOAMDataloader` verified earlier via
the GitHub source. If you'd rather install from source: the git URL used
during development, `github.com/FlowModelingControl/flowtorch`, may now
redirect to a renamed org (`AndreWeiner/flowtorch`) -- prefer the PyPI
package above unless you specifically need an unreleased commit.

Then get the actual dataset (a separate download from the library itself --
`flowtorch.DATASETS` is populated by scanning a local directory, not a
built-in remote registry, see `data_loading.py`'s docstring) and point
flowTorch at it:

```
export FLOWTORCH_DATASETS=/path/to/flowtorch/datasets
python -c "from flowtorch import DATASETS; print('of_cylinder2D_binary' in DATASETS)"
```

That must print `True` before `run_phase1.py` will do anything real. In
this environment the dataset lives at
`/mnt/s/opencode/sage-cfd/data/datasets_29_10_2021/datasets` (see
`../PATH_FORWARD.md` §3 for the exact environment used to produce
`results_v2/`).

**The real dataset is available and has been run.** `of_cylinder2D_binary`
is flowTorch OpenFOAM.com v2006 `pimpleFoam` (laminar), Re=100, 400
snapshots, 13,678 cells full domain / 7,190 in the ROI. Results below are
real, not scaffolded.

```
pip install torch
python run_phase1.py
```

## The corrected protocol, in brief

The original protocol (rollout from t=0.025, `s`-based energy, float32)
produced results that did not survive scrutiny -- see `phase1_report.md`
§5 for the full list of corrections. The current protocol
(`PATH_FORWARD.md` Step 1, implemented in `run_phase1.py::run_v2`):

- Every DMD forecast restarts from the **last training snapshot**, not
  from the start of the trajectory, and is scored over the held-out test
  window (81 steps here).
- Rolling-origin checks at a fixed horizon (H=20) from four origins, so no
  conclusion rests on a single start point.
- Three reference predictors are always reported alongside any model: the
  training mean, persistence, and the per-window POD(train) projection of
  the test data (the floor for any model in that basis).
- Error is reported both as full-field relative L2 and normalised by the
  fluctuation about the training mean -- the latter separates "predicting
  the mean" from "predicting the dynamics."
- POD rank/energy uses **s² (singular value squares)**, not `s`. Under the
  old `s`-based convention the 99%-energy rank on the full window was
  reported as 63; under `s²` it is **11** (99.9% is 23). Use the sweep
  table in `phase1_report.md`, not a single energy-threshold rank, when
  choosing a rank -- conjugate eigenvalue pairs can straddle an
  energy-threshold cutoff (see the r8/r9 rank-parity note there).
- Fit windows are swept explicitly: `full` (t<8) and `post_transient`
  (4<=t<8).
- Data efficiency uses the **last** n snapshots before the split, not the
  first n (the first n is pure startup transient).
- Outputs go to `phase1/results_v2/` (float64, canonical) and
  `phase1/results_v2_float32/` (float32, precision provenance only). The
  legacy protocol remains available via `run_phase1.py --legacy` and
  writes to the old `results/`-style directories, which are now
  historical.

See `phase1_report.md` for the full sweep tables and conclusions, and
`../PATH_FORWARD.md` for the decision record this protocol implements.

## float64 requirement

flowTorch's `SVD`/`DMD` classes go through a Gram-matrix eigendecomposition
path rather than a direct SVD of the data matrix. In float32, once the
singular-value ratio `s_r/s_1` drops below roughly 1e-3, this path loses
mode orthogonality (`||UᵀU - I||_F` reaches order 1-5 at rank 23-63), and
the resulting DMD operator is measurably wrong -- errors that look like
"39 unstable eigenvalues at |λ|=1.07" or a 6.03e9 rollout blowup in float32
become 10 unstable eigenvalues at |λ|≈1.002 and a 7.45% rollout error in
float64, on the identical fit. **Always run in float64** --
`load_cylinder_snapshots(dtype=torch.float64)` is the default. See
`phase1_report.md` §6 for the full precision comparison table, and
`check_precision.py` for the diagnostic that traces this.

## What's real vs. scaffolded

**Verified correct, with real bugs found and fixed along the way** (first
against a synthetic fixture with known ground truth -- `test_synthetic.py`
-- and now confirmed against the real dataset):

- `pod_baseline.py` -- POD reconstruction error decreases monotonically
  with rank, as it must; confirmed against `flowtorch.analysis.SVD`
  directly. Cumulative energy is computed from `s²`, not `s` (fixed --
  the earlier `s`-based version silently mis-selected the 99%-energy
  rank; see `phase1_report.md` §5).
- `dmd_baseline.py` -- **the important one to read the module docstring
  for.** `flowtorch.analysis.DMD`'s `.reconstruction`/`.dynamics`
  properties are NOT reliable in the installed version (flowtorch
  1.6.1): verified with a from-scratch exact linear system that
  `.predict()` recovers the true trajectory to 1e-6 relative error while
  `.reconstruction` on the same fit is off by 139%. Traced into the
  library source: `dynamics` builds its Vandermonde matrix with legacy
  `torch.vander` (descending powers), `.predict()` uses
  `torch.linalg.vander` (increasing powers) -- two incompatible
  conventions in the same class. This module uses `.predict()`
  exclusively and documents why; **do not use `dmd.reconstruction` as a
  correctness reference for anything**, in this scaffold or elsewhere,
  until/unless you verify it differently.
- `test_synthetic.py`'s fixture itself needed a real fix: an earlier
  version built oscillating modes as a bare `spatial_pattern(x) *
  cos(wt)` term with no companion `sin(wt)` term carrying an independent
  spatial pattern. That's not the trajectory of any autonomous linear
  system -- the instantaneous state doesn't contain enough information to
  determine its own next step -- so DMD correctly failed to fit it
  (recovered eigenvalues as low as |lambda|=0.06, wrong frequencies). Not
  a flowtorch bug, not a forecast() bug -- a fixture that wasn't actually
  DMD-representable. Fixed by pairing each frequency with proper cos/sin
  spatial-pattern pairs; DMD now recovers all five true eigenvalues
  exactly on the unit circle at the exact injected frequencies.
- `run_phase1.py`'s data-efficiency curve needed two fixes: (1) it
  originally reused the rank selected from the *full* training set
  (capped by `n-1`) when fitting on small subsets -- e.g. fitting a
  90-dimensional linear operator off 19 transitions at n=20, producing an
  extrapolation error of 461 at n=50 -- fixed by reselecting rank
  per-subset; (2) it used the *first* n snapshots, which for small n is
  pure startup transient and not informative about data efficiency on the
  target regime -- fixed by using the *last* n snapshots before the
  split (`PATH_FORWARD.md` §1 issue 6).
- `neural_rom_baseline.py` -- wiring was confirmed correct (gradients
  reach every parameter, loss decreases substantially with more
  training), but its extrapolation error (35.3%) was worse than the
  training-mean predictor (29.3%) once scored under the corrected
  protocol -- not a credible baseline. Dropped on this branch (see
  Scope, above); code retained but parked behind `--legacy --with-neural`.

**Real dataset now run and validated**: everything downstream of
`load_cylinder_snapshots()` has been exercised end-to-end against
`of_cylinder2D_binary`; results are in `phase1/results_v2/` (current) and
`phase1/results/`, `results_ini/`, `results_dmdpod/` (historical, flawed
protocol and/or float32 -- see `phase1_report.md` §5-6 before citing
numbers from those directories).

## Once you've run this

Compare the sweep table's DMD rows against the POD projection floor
(`phase1_report.md` §4) -- on the developed (post-transient) regime, DMD
sits within 1.2-2.3x of the floor at ranks >=9, essentially the best any
model in that POD basis can do. Full details, corrections to the earlier
(2026-09-05) report, and the precision investigation are in
`phase1_report.md`; the decision this closes (nonlinear DMD not needed for
this dataset) is recorded in `../PATH_FORWARD.md` §4.

Next: hand `t_split` and the resulting snapshot windows off to Phase 2's
Arm 2 (hand-specified) and Arm 3 (solver-derived) -- see
`../PATH_FORWARD.md` §2 Step 4 for how the static test is structured.
