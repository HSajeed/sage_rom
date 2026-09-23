# Step 5: Pre-registered decision rule for Step 4b

**Status:** DRAFT, awaiting owner sign-off. Once committed, this file must not change after any Step 4b data exists. The commit timestamp is the pre-registration record.

The thresholds below come from the Step 4a control (`results_step4a/noise_band.json`, commit `d88e106`). No 4b data has been produced or looked at.

## Question
Does parsed structure (Arm 3: textbook terms plus terms found by parsing the modified solver) beat hand-specified structure (Arm 2: `c + Az + H(z⊗z)`)? The test runs on data from a solver with an injected out-of-span term, `fvm::Sp(cD*mag(U), U)`, which is a −cD|U|U drag.

## Fixed protocol
Everything is identical to Step 4a unless listed here.
- **Data and split.** 4b drag case: same mesh, ROI (7,190 cells), write interval 0.025, and split t_split = 8.
- **POD.** float64, centred per window.
- **Fitting.**
  - Ridge penalties are chosen by the same 7×7 λ grid, the same 80/20 contiguous split and the same 20-step validation rollout.
  - The drag columns share λ_quad.
  - No re-tuning of the grid, ranks, windows or selection after 4b data is seen.
- **Arm 2.** Families {const, lin, quad}.
- **Arm 3.** Built by `arm3_library` from the parsed modified `UEqn.H` plus the case activity. It must include the `quadratic_drag` family Φᵀ(|Ũ|Ũ). If the parser does not produce it, the run is **invalid**, not negative.
- **Decisive cells.** Discrete (one-step map) target, **full window** (0 < t < 8, transient-including), ranks {8, 11, 15, 23}.

## Metrics
- `R_val = val_residual(Arm 2) / val_residual(Arm 3)`: the ratio of validation-block one-step residuals.
- `R_fc = forecast_full_field(Arm 2) / forecast_full_field(Arm 3)`: the 81-step forecast from t = 7.975.
- `floor_ratio(Arm 2)`: Arm 2's forecast error divided by the POD projection floor.

## Noise bands (from 4a, discrete target, full window)
- **Validation-residual band.** Take the max/min spread of `val_residual` over three sources: the base fit, leave-one-block-out (5 blocks), and λ×10 and λ/10 on each penalty.
  - By rank (r = 8, 11, 15, 23): 2.29, 1.82, 2.94, 2.19.
  - **Threshold `B_val = 3.0`**: the maximum, rounded up.
- **Forecast band.** Take the leave-one-block-out max/min spread of `forecast_full_field`, over all discrete cells in both windows.
  - Maximum: 1.16.
  - **Threshold `B_fc = 1.16`**.
- The forecast band excludes the λ×10 neighbours, which moved the forecast by up to 9× at r = 8. That measures sensitivity to λ selection, which both arms share; it is not noise in the Arm 2 vs Arm 3 comparison.

## Decision rule
Arm 3 **beats** Arm 2 if, at **≥ 3 of the 4 ranks**, both of these hold:
1. **Residual criterion.** `R_val > B_val = 3.0`.
2. **Forecast criterion.** Which test applies depends on `floor_ratio(Arm 2)`:
   - If `floor_ratio(Arm 2) ≥ 1.2` (Arm 2 has room above the floor): `R_fc > B_fc = 1.16`.
   - If `floor_ratio(Arm 2) < 1.2` (floor-limited, as it is at every rank in 4a): non-inferiority only, `R_fc ≥ 1/1.16`. The forecast cannot discriminate here (PATH_FORWARD §4).

**Arm 3 does not beat Arm 2** (a negative result) if the criteria hold at ≤ 1 rank.

**Inconclusive** covers everything else, including exactly 2 ranks or a placebo failure (below).

## Controls that must pass for any conclusion
1. **Null control on the 4b baseline run** (the unmodified solver, rerun alongside the drag case): Arm 3 ≡ Arm 2 exactly, as in 4a.
2. **Placebo (capacity) control.** Run Arm 2 + `quadratic_drag` (`--placebo-drag`) on the **4b baseline run**, i.e. the unmodified solver.
   - It must *not* meet the residual criterion (`R_val ≤ 3.0` at ≥ 3 ranks).
   - If it does, the extra regressor's capacity alone explains any gain, and the 4b result is **inconclusive**.
3. **Sanity checks.** The DMD reproduction and POD floor checks from 4a must pass on the 4b baseline run's own Phase 1 references. If those references don't exist, the check is recorded as not applicable.
4. **No divergence** in any decisive cell for either arm. A divergent Arm 2 cell with a non-divergent Arm 3 counts as Arm 3 meeting the forecast criterion at that rank. The reverse counts as failing it.

## Pre-run design requirement (detectability)
Before any 4b simulation, run `python -m static_rom.detectability` for the chosen injected term and amplitude, on the unmodified dataset. The injected out-of-span one-step signal must exceed `B_val × val_residual(Arm 2)` at ≥ 3 of the 4 decisive ranks. Otherwise the design can only produce a negative, so it must not be run.

**Status:** `quadratic_drag` at cD = 0.3 **fails** (signal/threshold 2e-4 to 3e-3 on the full window; `case_4b/detectability_quadratic_drag_cD0.3.json`). An injected term that passes this check is still to be chosen.

## Reported, non-decisive
- The continuous (finite-difference derivative) target. In 4a its λ selection hits the grid boundary and it sits 4–200× above the floor.
- The post_transient window, which is floor-limited.
- The fitted drag coefficient compared with the true `cD`: `−dt·cD` in the discrete map, projected. This is a plausibility check only.
- The fluctuation-normalised errors.
