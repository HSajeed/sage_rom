# SAGE-CFD -- Phase 1: conventional ROM baselines

POD, DMD, and a black-box neural ROM, all evaluated on the same task: fit
on an early time window of the cylinder-wake trajectory, forecast the held-
out later window. This is Phase 2's Arm 1 (`NeuralROM`) plus the classical
baselines Arm 2/Arm 3 need to beat.

## Setup (you run this part)

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

That must print `True` before `run_phase1.py` will do anything real.

```
pip install torch
python run_phase1.py
```

## What's real vs. scaffolded

**Verified correct, with real bugs found and fixed along the way** (all
against a synthetic fixture with known ground truth -- see
`test_synthetic.py` -- since the real dataset isn't available in this
environment):

- `pod_baseline.py` -- POD reconstruction error decreases monotonically
  with rank, as it must; confirmed against `flowtorch.analysis.SVD`
  directly.
- `dmd_baseline.py` -- **the important one to read the module docstring
  for.** `flowtorch.analysis.DMD`'s `.reconstruction`/`.dynamics`
  properties are NOT reliable in the installed version: verified with a
  from-scratch exact linear system that `.predict()` recovers the true
  trajectory to 1e-6 relative error while `.reconstruction` on the same
  fit is off by 139%. Traced into the library source: `dynamics` builds
  its Vandermonde matrix with legacy `torch.vander` (descending powers),
  `.predict()` uses `torch.linalg.vander` (increasing powers) -- two
  incompatible conventions in the same class. This module uses
  `.predict()` exclusively and documents why; **do not use
  `dmd.reconstruction` as a correctness reference for anything**, in this
  scaffold or elsewhere, until/unless you verify it differently.
- `neural_rom_baseline.py` -- wiring confirmed correct (gradients reach
  every parameter, loss decreases substantially with more training). Also
  confirmed a real *characteristic*, not a bug: low training loss does not
  imply accurate full-length rollout, because the rollout loss only ever
  supervises `rollout_horizon` (default 10) steps at a time. See the
  module docstring before assuming more epochs alone fixes a bad
  extrapolation number.
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
- `run_phase1.py`'s data-efficiency curve needed a fix too: it originally
  reused the rank selected from the *full* training set (capped by
  `n-1`) when fitting on small subsets -- e.g. fitting a 90-dimensional
  linear operator off 19 transitions at n=20, producing an extrapolation
  error of 461 at n=50. Fixed by reselecting rank per-subset. The curve is
  still non-monotonic in places even after that fix -- documented as a
  known fragility of energy-threshold rank selection on small noisy
  subsets, not something silently smoothed over. Look at which rank got
  picked at each point, and average over multiple seeds, before trusting
  this curve on real data.

**Not yet real:** everything downstream of `load_cylinder_snapshots()`,
since that needs the actual dataset. `run_phase1.py`'s orchestration logic
itself has been run end-to-end against the synthetic fixture and works;
only the real-data path is untested here.

## Once you've verified this against real data

Compare the printed table's `DMD` and `NeuralROM` rows -- DMD should
substantially beat the black-box neural ROM on both reconstruction and
extrapolation for this kind of clean, periodic, low-dimensional flow. If it
doesn't, that's worth understanding before moving on to Phase 2's arms,
not a result to wave past.

Next: hand `t_split` and the resulting snapshot windows off to Phase 2's
Arm 2 (hand-specified) and Arm 3 (solver-derived) once you're ready to
train those on the unmodified solver's data.
