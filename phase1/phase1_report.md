# Phase 1 Results Report — sage-cfd

*Updated: 2026-09-05 · Real-data validation of conventional baselines on the flowTorch cylinder2D dataset, now on the transient-including default window. The 2026-09-01 report (post-transient window only) is preserved in `phase1/results_ini/phase1_report.md`.*

## 1. Executive Summary

Phase 1 validates the conventional ROM baselines (POD, DMD, NeuralROM) against the real cylinder2D dataset. The dataset changed on 2026-09-05: the default snapshot window now **includes the transient** (t=0.025 → 10.0s, **400 snapshots**, `t_min=None` in `data_loading.py`), replacing the tutorials' post-transient-only window (t≥4.0s, 241 snapshots). This was a deliberate harder test: the transient growth phase is where nonlinear, non-periodic structure matters, and validating baselines there answers, before Phase 2, whether the flow regime rewards nonlinear ROMs at all.

Two runs have now been validated:

- **Post-transient window (2026-09-01, `results_ini/`)**: DMD achieved **0.26% extrapolation error** at rank 15 — the periodic, low-rank structure (Re=100) is nearly ideal for linear DMD. NeuralROM failed to extrapolate (59–60%).
- **Transient-including window (2026-09-05, `results/`)**: rank-63 (99% energy). **DMD catastrophically blows up — extrapolation error 6.03e9** — because the transient-growth phase injects eigenvalues off the unit circle (|λ|≈1.07, ~2.9/s growth), which the linear operator then amplifies exponentially over the 81-step test horizon. **NeuralROM improves to 0.353 extrapolation error** (rollout horizon 30, 800 epochs, ~2× training data) and its data-efficiency curve is now monotonically improving with data (0.417→0.345). POD reconstructs to 0.105% at rank-63.

The headline: **the transient-including window flips the conventional baseline picture.** Linear DMD is essentially perfect on the post-transient periodic regime and catastrophically unusable on the full window, while the nonlinear NeuralROM arm gracefully degrades. This is exactly the "hard test case" signal Phase 2's structured nonlinear arms are meant to exploit.

## 2. Dataset Description

- **Dataset**: flowTorch `of_cylinder2D_binary` — OpenFOAM (pimpleFoam) simulation
- **Benchmark**: Schäfer–Turek 2D confined cylinder, **Re=100** (D=0.1, U_mean=1.0, ν=1e-3)
- **Mesh**: 13,678 cells full domain; **7,190 cells** in ROI window [0.1, −1] → [0.75, 1]
- **State variable**: velocity **U**, stacked `[u_x; u_y]` → data matrix `(14,380, N)`
- **Total snapshots**: 401 (t=0 → 10s, dt=0.025s)
- **Transient-including window** (default, `t_min=None`): **400 snapshots** (t=0.025 → 10.0s)
- **Train/test split** at t=8.0s: **319 train / 81 test** (held-out temporal extrapolation)
- Post-transient window (t ≥ 4.0s, `t_min=4.0`, opt-in for reproducing the old result): 241 snapshots, 160 train / 81 test

Note: the ~0.164 Hz "expected Strouhal" that applies to free-stream cylinder flows does **not** apply here. This is a channel-confined Schäfer–Turek geometry; the correct shedding frequency is **~3.0 Hz** (f = St·U/D = 0.2978·1.0/0.1). The dataset's own settings (radius 0.05, channel height 0.41, nu 1e-3) confirm this.

## 3. Methodology

| Baseline | Role | Metrics |
|----------|------|---------|
| POD | Static spatial basis (no forecast) | Reconstruction error vs. rank; fit time |
| DMD | Linear dynamics in POD basis | Reconstruction + extrapolation error; fit time |
| NeuralROM | Black-box latent dynamics (Phase 2 "Arm 1") | Reconstruction + extrapolation error; fit time |

Shared settings (transient-including run):
- Rank = 63 (99% cumulative POD energy on 319 training snapshots; was 15 on the post-transient window)
- DMD delegates to `flowtorch.analysis.DMD.predict()` (the documented Vandermonde bug makes `.reconstruction` unreliable)
- NeuralROM: linear encoder/decoder + 3-layer tanh MLP, latent_dim=8, **800 epochs**, short-rollout supervision (**horizon 30**), CUDA auto-selected when available
- Data efficiency: extrapolation error vs. training-size ∈ {20, 50, 100, 319}, rank re-selected per subset

## 4. Results — Transient-Including Window (current default)

### 4.1 Comparison Table

| Baseline | Reconstruction Err | Extrapolation Err | Fit Time (s) |
|----------|-------------------|-------------------|--------------|
| POD | 0.00105 (0.105%) | n/a (static basis) | 0.063 |
| **DMD** | 9.23e6 | **6.03e9** (blowup) | 0.149 |
| NeuralROM | 0.352 (35.2%) | **0.353 (35.3%)** | 322.5 |

### 4.2 POD Reconstruction Error vs. Rank

| Rank | Relative Error |
|------|---------------|
| 1 | 0.212 (21.2%) |
| 2 | 0.127 (12.7%) |
| 4 | 0.069 (6.9%) |
| 63 | 0.00105 (0.105%) |

The transient-window POD basis is far more expensive than the post-transient one: rank-2 (the dominant conjugate shedding pair) captures only 87% of the energy when the transient is included, versus rank-4 reaching 3.6% error on the post-transient window. Rank 63 (99% energy) reconstructs to ~0.1%.

### 4.3 DMD Eigenvalue Analysis

- **Rank**: 63 (99% energy threshold on 319 training snapshots)
- **63 eigenvalues: 39 with magnitude > 1.0 (growing), 24 decaying, 0 exactly on the unit circle.** Only 22 lie within [0.99, 1.01] (near-unit).
- **Fastest-growing modes**: |λ| = 1.0747 at **f = ±1.646 Hz** and |λ| = 1.0734 at **f = ±5.219 Hz**, both with continuous-time growth rate **~2.88/s**; several more at ~2.5/s
- **Fastest-decaying modes**: |λ| as low as 0.46 (growth −30.9/s) — the transient startup modes DMD correctly identifies or spuriously assigns
- Growing modes amplify over the 81-step (2.025s) test window: 1.0747^81 ≈ **340× per-mode amplitude growth**, compounded across ~20+ growing modes → the 6.03e9 extrapolation error

Interpretation: the linear DMD operator **cannot represent transient growth → nonlinear saturation**. It fits the observed growth phase as genuinely unstable eigenvalues (brute-force linear least squares), then forecasts exponential blowup instead of the saturated vortex street. This is the classic limitation of linear Koopman/DMD approximations on non-normal, nonlinearly saturating flows (see `future.md` §3).

### 4.4 Data Efficiency (extrapolation error vs. training snapshots)

| Training Snapshots | DMD | NeuralROM |
|--------------------|-----|-----------|
| 20 | 0.360 | 0.417 |
| 50 | 10.35 | 0.407 |
| 100 | 3.40 | 0.394 |
| **319** | **6.03e9** | **0.345** |

- **DMD**: degrades monotonically as more (transient) data is added — n=319 (full transient) is catastrophic. Rank re-selection per n (13/20/51/63) shifts which unstable modes are captured, so intermediate sizes land on different (still unstable) subsets.
- **NeuralROM**: monotonically improves with data — 0.417 → 0.345 with n=20→319. This is the first time NeuralROM shows a clean data-scaling signal on this dataset (the old post-transient run was flat ~63% regardless of n).

### 4.5 Rank Selection Stability

| Training Snapshots | Rank (99% energy) |
|--------------------|-------------------|
| 20 | 13 |
| 50 | 20 |
| 100 | 51 |
| 160 | 63 |

The transient window needs many more modes at high energy (rank jumps 13→63 as data grows), which is itself informative: the transient adds a broad spectral content that a periodic-only window does not have. Rank selection remains unstable at small n (documented caveat in `status.md` §8.4).

## 5. Physical Validation

| Check | Post-transient | Transient-including |
|-------|----------------|---------------------|
| Re=100 confirmed (D=0.1, U=1.0, ν=1e-3) | ✅ PASS | ✅ PASS |
| Shedding frequency 3.0 Hz = f·D/U (St≈0.298) | ✅ PASS | ✅ PASS (dominant near-unit modes retain ~3 Hz) |
| DMD eigenvalues on unit circle | ✅ PASS | ❌ FAIL (39 growing, 24 decaying) |
| DMD conjugate pairs present | ✅ PASS | ✅ PASS (growing/decaying modes appear as ±f pairs) |
| POD error decreases monotonically with rank | ✅ PASS | ✅ PASS |
| DMD extrapolation error low | ✅ PASS (0.26%) | ❌ FAIL (6.03e9) |
| NeuralROM extrapolation | ❌ FAIL (59%) | ⚠️ PARTIAL (35% — improved, not yet credible) |
| DMD data-efficiency monotonicity | ❌ FAIL | ❌ FAIL (gets worse with more data) |

## 6. Known Limitations

1. **DMD rank-selection fragility at small n** — non-monotonic data-efficiency curve; single-seed, single-sample per n.
2. **Linear DMD cannot represent transient growth → saturation** — on the transient-including window this is a *category-level* failure (6.03e9 extrapolation), not a tuning issue. Fixed-rank low-order DMD on the transient window is untested (see Outstanding Tasks) — it may filter growing high-rank modes but would discard the transient physics entirely.
3. **NeuralROM long-horizon accuracy still weak** — 35% extrapolation error is a big improvement over 59% but not yet a credible baseline; single seed (run-to-run variance was 0.59–0.67 on the old config, needs re-measuring on the new config). No loss-curve/convergence tracking. Latent dim fixed at 8.
4. **Single trajectory, single Reynolds number** — "generalization" here means temporal extrapolation only.
5. **NeuralROM timing on GPU** (322s fit incl. overhead): training still uses short-rollout loss (horizon 30) — teacher-forcing limitation remains, reduced but not eliminated.

## 7. Conclusions & Implications for Phase 2

- **The conventional-baseline picture depends entirely on the window.** On the post-transient periodic regime, linear DMD is near-perfect (0.26%) and a hard bar to beat. On the full transient-including window — the new default — linear DMD is unusable for forecasting (6.03e9) while nonlinear NeuralROM degrades gracefully to 35%.
- **This directly motivates the Phase 2 thesis**: the transient regime is exactly where nonlinearity matters (growth → saturation), and where a nonlinear structured ROM (operator-composable, KPCA-style, or deep-Koopman) has the potential to beat both the linear baseline (which blows up) and the black-box baseline (which is merely mediocre).
- **DMD extension candidates are documented in `future.md`**: time-delay/Hankel DMD (HODMD) and kernel DMD are the cheapest drop-in fixes to test on this window; physics-informed DMD (piDMD) and bilinear/quadratic-closure DMD are the medium tier; deep Koopman autoencoders are deferred as overkill at Re=100.
- **NeuralROM (Arm 1) needs multi-seed statistics** (and ideally convergence tracking) before it is a credible black-box baseline against arms 2/3 — the current 35% is a single-sample estimate.
- The data-efficiency comparison should re-fit with fixed ranks per n (or average over seeds) before drawing "data-hunger" conclusions.
- All results and metadata are persisted in `phase1/results/` (`full_results.pkl`, per-metric JSON/CSV files, `run_metadata.json`). Old post-transient artifacts remain in `phase1/results_ini/`.