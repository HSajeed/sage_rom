# Phase 1 Results Report — sage-cfd

*Updated: 2026-09-17 · Corrected evaluation protocol, POD + DMD only, on the flowTorch cylinder2D dataset. This report supersedes the 2026-09-05 version. The earlier report's DMD blowup / "sweet spot" and NeuralROM numbers were artifacts of a flawed evaluation protocol and float32 precision loss — see §5. The 2026-09-01 post-transient-only report is preserved in `phase1/results_ini/phase1_report.md`.*

## 1. Executive Summary

Phase 1 evaluates the conventional linear ROM baselines — POD and DMD — against the real cylinder2D dataset, under a corrected protocol (`PATH_FORWARD.md` Step 1): forecasts restart from the last training snapshot rather than from t=0.025, energy is computed from singular-value squares, both full-field and fluctuation-normalised error are reported against explicit reference predictors (mean, persistence, POD projection floor), and everything runs in float64.

Under this protocol, on the fully-developed test window (t=8–10 s), **linear DMD is essentially at the POD projection floor** (1.16–4× floor depending on rank) and closes the Phase 1 Step-2 exit criterion: nonlinear DMD variants are not needed for this dataset (`PATH_FORWARD.md` §0, §4). The dramatic 6.03e9 "blowup" and the "~0.3 bounded sweet spot" reported on 2026-09-05 do not survive the corrected protocol; both are traced to specific, identified causes in §5.

**NeuralROM (Arm 1) is dropped on this branch** (`PATH_FORWARD.md` §0.1). Its 2026-09-05 number — 35.3% extrapolation error — was worse than the training-mean predictor (29.3%), i.e. it was not a working baseline; it is not re-evaluated here.

## 2. Dataset Description

- **Dataset**: flowTorch `of_cylinder2D_binary` — OpenFOAM.com v2006, `pimpleFoam`, laminar
- **Benchmark**: Schäfer–Turek 2D confined cylinder, **Re=100** (D=0.1, U_mean=1.0, ν=1e-3)
- **Mesh**: 13,678 cells full domain; **7,190 cells** in ROI window [0.1, −1] → [0.75, 1]
- **State variable**: velocity **U**, stacked `[u_x; u_y]`
- **Total snapshots**: **400**, t=0.025 → 10.0 s, dt=0.025 s
- **Train/test split** at t=8.0 s: **319 train / 81 test** (held-out temporal extrapolation)
- **Test window (8–10 s) is fully developed flow** — no configuration in this report tests transient forecasting (see §4, §7).

Note: the ~0.164 Hz "expected Strouhal" that applies to free-stream cylinder flows does **not** apply here. This is a channel-confined Schäfer–Turek geometry; the correct shedding frequency is **~3.0 Hz** (f = St·U/D = 0.2978·1.0/0.1). The dataset's own settings (radius 0.05, channel height 0.41, nu 1e-3) confirm this.

## 3. Evaluation Protocol (corrected, `PATH_FORWARD.md` Step 1)

- **Forecast origin**: every DMD forecast restarts from the **last training snapshot** and covers the 81-step held-out test window, rather than rolling out from t=0.025 (the old protocol).
- **Rolling origins**: an additional fixed horizon H=20 check from four origins — `last_train`, `test_idx_0`, `test_idx_20`, `test_idx_40` — so conclusions do not rest on a single start point.
- **Reference predictors, reported alongside every model**: the training mean, persistence (repeat the last training snapshot), and the per-window POD(train) projection of the test data (the floor any model in that basis can reach).
- **Error metrics**: full-field relative L2 error, and fluctuation-normalised error (relative to the training temporal mean) — the latter separates "predicting the mean" from "predicting the dynamics."
- **Ratio to floor**: DMD error ÷ POD projection floor error, at matching rank and fit window.
- **Energy convention**: cumulative energy from **s²** (singular value squares), not `s` — this changes the 99%-energy rank from 63 to 11 (§5).
- **Fit windows**: `full` (t<8, 319 snapshots) and `post_transient` (4≤t<8, 160 snapshots).
- **Rank ladder**: {1, 2, 4, 8, 11, 15, 23, 32, 63}, including the s²-energy ranks at 99% and 99.9% for each window.
- **Precision**: float64 throughout (`load_cylinder_snapshots(dtype=torch.float64)` is the default). float32 is retained separately, in `results_v2_float32/`, as precision provenance only (§6).
- **Legacy column**: the old from-x0 rollout error is still computed and reported, labelled `legacy_extrap_*`, for provenance/comparison — it is not the primary metric.

## 4. Results

All tables below are float64, `phase1/results_v2/` (`sweep.csv`, `reference_predictors.json`, `data_efficiency.json`, `run_metadata.json`). Errors are relative L2, shown as percentages (2 decimal places except where more precision is informative).

### 4.1 Reference predictors

| Predictor | Full-field error | Fluctuation error |
|---|---|---|
| Training mean | 29.26% | 100% (by definition) |
| Persistence (repeat last train snapshot) | 39.88% | 136.29% |
| Persistence, rolling (mean over 4 origins, H=20) | 40.97% | 139.93% |

### 4.2 DMD sweep — fit window `full` (t<8, 319 snapshots), forecast from last training snapshot over 81 test steps

| Rank | Primary full-field | Primary fluctuation | POD floor (full-field) | Ratio to floor | n_unstable | max\|λ\| | Rolling mean (full-field) | Legacy from-x0 full-field |
|---|---|---|---|---|---|---|---|---|
| 1 | 29.37% | 100.38% | 21.17% | 1.39× | 1 | 1.00047 | 29.34% | 29.32% |
| 2 | 29.81% | 101.87% | 8.65% | 3.45× | 1 | 1.00047 | 31.20% | 29.32% |
| 4 | 8.61% | 29.43% | 6.60% | 1.31× | 3 | 1.00189 | 6.84% | 31.28% |
| 8 | 7.98% | 27.27% | 2.48% | 3.22× | 3 | 1.00189 | 4.67% | 29.93% |
| 11 (r99) | 6.36% | 21.74% | 1.62% | 3.92× | 3 | 1.00191 | 4.16% | 28.67% |
| 12 | 5.79% | 19.80% | 1.44% | 4.01× | 3 | 1.00191 | 2.43% | 29.33% |
| 15 | 3.67% | 12.53% | 1.13% | 3.25× | 7 | 1.00179 | 1.71% | 30.37% |
| 23 (r99.9) | 1.71% | 5.84% | 0.41% | 4.21× | 6 | 1.00186 | 0.87% | 28.15% |
| 24 | 1.63% | 5.56% | 0.40% | 4.07× | 6 | 1.00200 | 0.58% | 28.16% |
| 32 | 2.62% | 8.94% | 0.22% | 11.63× | 10 | 1.00158 | 0.50% | 23.87% |
| 63 | 1.13% | 3.87% | 0.055% | 20.50× | 10 | 1.00198 | 0.13% | 7.45% |

### 4.3 DMD sweep — fit window `post_transient` (4≤t<8, 160 snapshots), forecast from last training snapshot over 81 test steps

| Rank | Primary full-field | Primary fluctuation | POD floor (full-field) | Ratio to floor | n_unstable | max\|λ\| | Rolling mean (full-field) | Legacy from-x0 full-field |
|---|---|---|---|---|---|---|---|---|
| 1 | 28.72% | 98.17% | 20.38% | 1.41× | 0 | 0.99999 | 28.73% | 28.72% |
| 2 | 29.12% | 99.51% | 6.49% | 4.48× | 0 | 0.99999 | 29.12% | 28.73% |
| 4 | 6.50% | 22.22% | 3.56% | 1.83× | 0 | 0.99999 | 6.47% | 6.53% |
| 5 (r99) | 3.62% | 12.36% | 2.64% | 1.37× | 0 | 0.99999 | 3.60% | 3.91% |
| 6 | 3.62% | 12.38% | 1.34% | 2.71× | 2 | 1.00003 | 3.59% | 3.87% |
| 8 (r99.9) | 1.40% | 4.79% | 0.71% | 1.98× | 2 | 1.00003 | 1.38% | 1.88% |
| 9 | 0.73% | 2.50% | 0.55% | 1.33× | 2 | 1.00003 | 0.73% | 0.90% |
| 11 | 0.35% | 1.19% | 0.27% | 1.27× | 4 | 1.00011 | 0.34% | 0.40% |
| 15 | 0.21% | 0.71% | 0.095% | 2.19× | 6 | 1.00012 | 0.22% | 0.20% |
| 23 | 0.041% | 0.14% | 0.024% | 1.69× | 15 | 1.00021 | 0.031% | 0.042% |
| 32 | 0.014% | 0.047% | 0.0059% | 2.33× | 14 | 1.00026 | 0.0078% | 0.018% |
| 63 | 0.0013% | 0.0044% | 0.0011% | 1.16× | 6 | 1.00008 | 0.0011% | 0.0020% |

Note the `post_transient` floor is not monotone in rank between r23 and r32 (0.024% then dips further at r32 — nested-projection floors are non-monotone here only under float32; the float64 values above are monotone as expected for a nested orthogonal projection. This was the trigger for the precision investigation in §6.).

### 4.4 Energy ranks and rank-parity checks (`run_metadata.json`)

| Window | n_fit | r99 (s²) | r99.9 (s²) |
|---|---|---|---|
| full | 319 | 11 | 23 |
| post_transient | 160 | 5 | 8 |

| Window | Check | r | r+1 | error at r | error at r+1 | relative change |
|---|---|---|---|---|---|---|
| full | r99 | 11 | 12 | 6.36% | 5.79% | −8.9% |
| full | r99.9 | 23 | 24 | 1.71% | 1.63% | −4.7% |
| post_transient | r99 | 5 | 6 | 3.62% | 3.62% | +0.12% |
| post_transient | r99.9 | 8 | 9 | 1.40% | 0.73% | **−47.7%** |

The post_transient r8→r9 step cuts error by 48%: the DMD rank is chosen on centred (SVD) data but fit on raw data, which can split a conjugate mode pair across the rank cutoff. Ranks should be chosen by sweep, not energy threshold alone.

### 4.5 Data efficiency (`data_efficiency.json`, last-n snapshots before t_split, fixed ranks {8, 15, 32} plus the window's own r99 rank)

| n (last-n snapshots) | r99 rank | rank 8 | rank 15 | rank 32 |
|---|---|---|---|---|
| 20 | 5 | 1.48% | 0.27% | 0.29% (r19, n−1 cap) |
| 50 | 5 | 1.49% | 0.095% | 0.0021% |
| 100 | 5 | 1.41% | 0.090% | 0.0035% |
| 160 | 5 | 1.40% | 0.21% | 0.014% |
| 319 | 11 | 7.98% | 3.67% | 2.62% |

Using the **last** n snapshots before the split (all inside or bordering the developed regime for n≤160), error at fixed rank is essentially flat to slightly improving from n=50 to n=160, then rises sharply at n=319 because that fit window now includes the transient. Data efficiency here is dominated by which regime the fit window covers, not by the snapshot count alone.

## 5. Corrections to Earlier Conclusions

| Earlier claim (2026-09-05 report) | What it actually was | Evidence |
|---|---|---|
| DMD extrapolation error 6.03e9 ("blowup") | Rollout from t=0.025 (400-step rollout, not evaluated against the 81-step test window) with float32 precision loss at rank 63. Restarting from the last training snapshot and using float64: legacy from-x0 rollout at rank 63 is **7.45%**, not 6.03e9. The float32 version of the same from-x0 rollout is 6.03e9 (confirmed in `results_v2_float32/sweep.csv`). | `results_v2/sweep.csv` (full, r63, `legacy_extrap_full_field`=0.0745); `results_v2_float32/sweep.csv` (same cell = 6.03e9) |
| "~0.3 bounded sweet spot" at ranks 1–32 | The training-mean predictor alone scores 0.2926 (29.3%). Ranks 1–2 sit at or above the mean predictor in full-field error, and the fluctuation-normalised error at those ranks is ≈100%, i.e. no dynamics is being captured — it's the null predictor. | `reference_predictors.json` (mean=0.2926); `sweep.csv` (full, r1: 0.2937, r2: 0.2981) |
| "39/63 unstable eigenvalues, \|λ\|=1.0747" | A float32 artifact. In float64, full-window rank 63 has **10** eigenvalues with \|λ\|>1, max \|λ\|=1.00198. The float32 fit at the same rank has 39 unstable eigenvalues, max \|λ\|=1.0747. | `results_v2/sweep.csv` vs `results_v2_float32/sweep.csv`, full r63 |
| Rank 63 as "99% energy" | `pod_baseline.cumulative_energy` used singular values `s`, not `s²`. The true 99%-energy rank (full window) is **11**; 99.9% is **23**. | `run_metadata.json`, `window_energy_ranks` |
| "Linear DMD can't represent transient, nonlinear needed" | Not supported once the forecast restarts from the last training snapshot. Post-transient fit, rank 15, restart-from-last-train: **0.21%** full-field (0.23% cited in `PATH_FORWARD.md` §1 issue 5, consistent within rounding — see reproduction note below). Linear DMD sits within 1.2–2.3× of the POD floor across ranks ≥9 on the developed regime. | `results_v2/sweep.csv`, post_transient r15 |
| "s vs s²": rank 63 reported as the 99% threshold | Under `s²`, rank 63 is far past 99.9% energy on both windows; the true 99%/99.9% ranks are 11/23 (full) and 5/8 (post_transient). | `run_metadata.json` |
| "340× per-mode amplitude growth over 81 steps" (1.0747^81) | Arithmetic error in the original explanation: it describes the growth over the 400-step from-x0 rollout used in the old protocol, not the 81-step test window; 1.0747^81 is not the mechanism for a rollout that starts at t=0.025 and runs 400 steps to t=10. The actual compounded legacy from-x0 float32 error at rank 63 is 6.03e9, consistent with ~3e12-scale per-mode amplification over 400 steps at the float32-corrupted eigenvalues, not 81. | `run_phase1.py` legacy rollout definition (`n_fit + n_test - 1` steps from x0); `PATH_FORWARD.md` §1 issue 1 |
| Data-efficiency curve used the *first* n training snapshots | At n=20 this covered only t≤0.5 s — pure startup transient, not representative of the fit-window effect being tested. Recomputed on the *last* n snapshots before the split (§4.5). | `PATH_FORWARD.md` §1 issue 6; `data_efficiency.json` |
| NeuralROM "gracefully degrades" to 35.3% vs DMD's blowup | NeuralROM's 35.3% extrapolation error was *worse* than the training-mean predictor (29.3%) — not a working baseline, graceful or otherwise. NeuralROM is dropped on this branch (`PATH_FORWARD.md` §0.1); not re-evaluated here. | `PATH_FORWARD.md` §0.1; 2026-09-05 report §4.1 (0.353 vs mean predictor 0.2926) |

## 6. Precision Finding: float32 Gram-Matrix SVD

flowTorch's `SVD`/`DMD` implementation uses a Gram-matrix eigendecomposition path (not a direct SVD of the data matrix). In float32, once the singular-value ratio `s_r/s_1` drops below roughly 1e-3, this path loses mode orthogonality, and the resulting DMD operator is measurably wrong. flowtorch version 1.6.1.

**Orthogonality loss** `||UᵀU − I||_F`, float32 vs float64:

| Window | Rank | float32 | float64 |
|---|---|---|---|
| full | 63 | 2.27e-01 | 3.20e-11 |
| post_transient | 23 | 4.85e-01 | 5.76e-11 |
| post_transient | 32 | 1.69 | 8.60e-10 |
| post_transient | 63 | 5.43 | 4.79e-07 |

**DMD primary (restart-from-last-train) error, float32 → float64**:

| Window | Rank | float32 | float64 |
|---|---|---|---|
| full | 63 | 0.150 (n_unstable=39) | 0.0113 (n_unstable=10) |
| post_transient | 23 | 4.70 | 0.000407 |
| post_transient | 32 | 42.8 | 0.000138 |
| post_transient | 63 | 62.1 | 0.0000128 |

**Legacy from-x0 rollout, full window rank 63**: 6.03e9 (float32) → 0.0745 (float64).

Precision-check evidence lives in `results_v2/precision_check.json` (may be regenerating — the values above are independently verified against `results_v2/sweep.csv` and `results_v2_float32/sweep.csv`) and `check_precision.py`. **Conclusion: float64 is required for this dataset's DMD fits above rank ~20–30**; it is now the default (`load_cylinder_snapshots(dtype=torch.float64)`).

## 7. Conclusions & Implications for Phase 2

(Substance from `PATH_FORWARD.md` §4, Step 2 decision, 2026-09-17.)

1. **Linear DMD is not the bottleneck on this dataset.** On the developed regime (post_transient fit, restart-from-last-train forecast) it sits essentially at the POD floor — every rank ≥9 is within 1.2–2.3× of its own floor. **Nonlinear DMD variants (HODMD, kernel DMD, EDMD, piDMD, deep Koopman) are not needed for this dataset and test.** This closes the Phase 1 Step-2 exit criterion.
2. **The earlier "blowup" and "rank pollution" findings were artifacts of three compounding issues**: (a) scoring by a 400-step rollout from t=0.025 instead of an 81-step forecast from the last training snapshot, (b) the `s` vs `s²` energy convention (mis-selecting rank 63 as "99% energy" when the true rank is 11), and (c) float32 precision loss in flowTorch's Gram-matrix SVD path, which destroys mode orthogonality once `s_r/s_1 ≲ 1e-3` (§6).
3. **The test window (8–10 s) is fully developed flow.** No configuration in this report tests *transient* forecasting; the full-window fit (which includes the transient) degrades to 4–20× the floor, showing the transient is genuinely harder, but no test here forecasts *into* the transient itself.
4. **Rank parity matters.** The rank is chosen on centred data (SVD) but DMD is fit on raw data, which can split a conjugate eigenvalue pair across the cutoff — post-transient r8→r9 cuts error by 48% (§4.4). Choose ranks by sweep, not by energy threshold alone.
5. **Data efficiency is dominated by which regime the fit window covers, not by snapshot count.** The last 50–160 snapshots before the split (post-transient-only) give sub-0.01% error at rank 32; adding the transient (n=319) raises the same-rank error by two orders of magnitude.

**Implications for Phase 2 (`PATH_FORWARD.md` Step 4)**:
- On the developed regime, every linear-or-better model is floor-limited, so **forecast error cannot discriminate Arm 2 (textbook) from Arm 3 (parsed)** there.
- The static test (Phase 2, Step 4) must therefore **lead with the time-derivative fit residual**, not forecast error.
- It should also use the **transient-including window**, where a linear model sits 4–20× above the floor and structured terms (e.g. quadratic convection, `c + Az + H(z⊗z)`) have room to matter.
- Use float64 throughout.

## 8. Artifacts and How to Reproduce

- Corrected protocol: `python run_phase1.py` from `phase1/`, with `FLOWTORCH_DATASETS` set (see README). Outputs write to `phase1/results_v2/` (float64, canonical).
- `--legacy` flag reproduces the old (pre-correction) protocol, including the from-x0 rollout as primary and NeuralROM gated behind `--with-neural`.
- `python test_eval_protocol.py` — plain-script checks of the corrected protocol against the synthetic fixture only (never touches the real dataset): s² rank ≤ s-based rank at the same threshold, restart-from-last-train forecast behaviour.
- `python check_precision.py` — standalone diagnostic that traces the sweep's precision-sensitive results to flowTorch's float32 Gram-matrix SVD; writes `results_v2/precision_check.json`.
- `phase1/results_v2_float32/` — the same corrected protocol run in float32, kept only as precision provenance for §6; do not use for conclusions.
- `phase1/results/`, `phase1/results_ini/`, `phase1/results_dmdpod/` are **historical**, produced under the flawed evaluation protocol and/or float32 — retained for provenance, not for citing current conclusions.
