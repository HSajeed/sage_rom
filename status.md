# Technical Report: sage-cfd

*Updated 2026-09-17. This revision supersedes the 2026-09-06 content below: the Phase 1 evaluation protocol was found to be flawed (rollout-from-t0, `s`-not-`s²` energy, float32 precision loss) and has been corrected; see `phase1/phase1_report.md` (rewritten 2026-09-17) for canonical Phase 1 numbers, and `PATH_FORWARD.md` (local, gitignored — not in git — for the authoritative running plan and decision log) for the full review and open items.*

## 1. Purpose

**SAGE-CFD** is a research scaffold investigating whether ROMs (reduced-order models) for CFD benefit from operator structure *derived from parsing solver source code* rather than hand-specified textbook physics.

**NeuralROM (Arm 1) is dropped on the `DMD_POD` branch** (`PATH_FORWARD.md` §0.1): the core question is **Arm 2 (hand-specified textbook incompressible-NS structure) vs Arm 3 (structure derived from AST-parsing OpenFOAM solver source)**, and a black-box Arm 1 is not needed to answer it. POD and DMD serve as the **linear reference baselines** for Phase 1, not as arms of the Phase 2 comparison.

Phase 2 will first give an answer with a **static, closed-form model** (POD projection / least-squares regression on POD coefficients, no gradient training, no random seeds) before any trained-ROM redesign is revisited (`PATH_FORWARD.md` §0.3, Step 4). Within that static test, **nonlinear DMD variants are closed as unnecessary for this dataset** — linear DMD on the developed regime sits at the POD projection floor (`PATH_FORWARD.md` §4; `phase1/phase1_report.md` §7).

The hypothesis under test: **Arm 3 beats Arm 2**, proving code-parsing adds structural information a textbook specification misses.

## 2. Architecture / Layout

```
sage-cfd/
├── phase1/                  # Conventional baselines (POD/DMD/neural ROM)
│   ├── data_loading.py      # flowTorch cylinder2D dataset loader
│   ├── pod_baseline.py      # POD fit + shared energy rank selection (s² energy)
│   ├── dmd_baseline.py      # DMD wrapper around flowtorch .predict()
│   ├── neural_rom_baseline.py  # encoder/MLP-dynamics/decoder (Arm 1, dropped as of DMD_POD — see §1)
│   ├── metrics.py           # relative L2, timing, data-efficiency harness
│   ├── run_phase1.py        # orchestration + comparison table; run_v2 = corrected protocol
│   ├── test_synthetic.py    # closed-form synthetic fixture w/ known frequencies
│   ├── test_eval_protocol.py  # checks of the corrected (v2) protocol against the synthetic fixture
│   ├── check_precision.py   # float32-vs-float64 DMD precision diagnostic
│   ├── results_v2/          # canonical corrected-protocol results (float64)
│   └── results_v2_float32/  # same protocol, float32 — precision provenance only, not for conclusions
└── phase2/
    ├── extractor/
    │   ├── ontology.py          # fixed fvm/fvc/fvi → physics vocabulary
    │   ├── ast_parser.py        # tree-sitter C++ extraction of DSL calls
    │   ├── fvschemes_parser.py  # regex parser for fvSchemes dicts
    │   ├── operator_graph.py    # fuses AST+fvSchemes(+dispatch) into networkx DiGraph
    │   ├── dispatch.py          # hand-curated virtual-dispatch expansion table (e.g. divDevReff -> Stokes)
    │   └── llm_labeler.py       # STUB: prompt template + schema, no API call
    ├── rom/composable_operator_rom.py  # single ROM class + Arm2/Arm3 builders (retained, not current test vehicle — see phase2/README.md)
    ├── validation/
    │   ├── ground_truth_icofoam.yaml            # REVIEWED 2026-08-27 (all 10 entries reviewed:true)
    │   ├── ground_truth_pimpleFoam_v2006.yaml    # DRAFT, 51 entries, all reviewed:false — needs human review
    │   ├── gate1_check.py                        # extractor precision/recall vs GT
    │   ├── check_pimplefoam_graph.py              # checks pimpleFoam v2006 UEqn assembly (dispatch expansion, sign/side)
    │   └── detect_modification.py                 # identity-based diff between sources
    └── fixtures/
        ├── icoFoam.C, fvSchemes            # real OpenFOAM-dev icoFoam.C + fvSchemes, plus two injected-modification variants
        └── pimpleFoam_v2006/                # real OpenFOAM.com v2006 pimpleFoam.C/UEqn.H/pEqn.H/... + PROVENANCE.txt
```

## 3. Component Interaction / Data Flow

**Phase 1 path:** `flowTorch dataset` → `load_cylinder_snapshots()` → `CylinderSnapshots` (dataclass, `(2·n_cells, n_snapshots)` matrix of stacked [uₓ; u_y], times, mask, dt) → `.split(t_split)` time-based train/test → fit POD / DMD / NeuralROM → `metrics.print_summary_table` + data-efficiency curves.

**Phase 2 path:** `icoFoam.C` → tree-sitter AST → `ExtractedCall` records (namespace, function, raw-text args, LHS variable via parent walk) → fused with `fvSchemes` entries (scheme per operator, OpenFOAM default-fallback) → networkx `DiGraph` with:

- nodes = DSL calls, attributes include `physical_type`, `confidence` (high/low/unlabeled), `scheme`
- `composition` edges: terms sharing an `assigned_to` equation target (e.g. UEqn)
- `data_flow` edges: a produced variable name appearing later as an argument (e.g. `phiHbyA`)
- hand-specified PISO skeleton stored in `graph["algorithm_skeleton"]` (explicitly NOT inferred)

Graph → `build_from_operator_graph()` maps distinct `physical_type`s to learnable blocks (`quadratic`/`linear`/`mlp`) → `ComposableOperatorROM` (latent dz/dt = Σ blocks, explicit Euler rollout).

## 4. Main Algorithms

- **Rank selection**: cumulative-energy threshold (99%) on POD singular values, shared between POD and DMD to avoid confounding basis-size differences.
- **DMD forecasting**: delegates entirely to `flowtorch.analysis.DMD.predict()` — deliberately avoids `.reconstruction`/`.dynamics`, documented as buggy (inconsistent Vandermonde conventions: legacy `torch.vander` descending powers vs `torch.linalg.vander` ascending; verified 139% error on an exact linear system while `.predict()` hit 1e-6).
- **Neural ROM**: linear encoder/decoder + 3-layer tanh MLP latent dynamics; loss = reconstruction + short-rollout (horizon 10, 8 random starts) MSE, Adam.
- **Composable ROM**: one `nn.Module` block per `OperatorSpec`; quadratic convection via `nn.Bilinear(z,z)`; explicit Euler integration; strict design constraint that Arms 2/3 share architecture, widths, and training — only `operator_specs` differ, each carrying auditable `provenance`.
- **AST extraction**: recursive tree-sitter walk; only `qualified_identifier` function nodes captured; `_find_enclosing_lhs` walks up unbounded through compositional node types to find `init_declarator`/`assignment_expression` LHS.
- **Modification detection**: diff by `(namespace, function, arguments)` identity tuple — immune to line drift.

## 5. Dependencies & Workflow

- Phase 1: `torch` + flowTorch (CFD library), **flowtorch 1.6.1**, installed and working in the `dmdsae` conda env. The PyPI package named `flowtorch` is an **unrelated Meta normalizing-flows project** — *verified against the PyPI registry*: summary "Normalizing Flows for PyTorch", Copyright Meta Platforms, source `facebookincubator/flowtorch`. **Note:** the phase1 README's install command (`git+https://github.com/FlowModelingControl/flowtorch.git`) has been updated to `pip install flowtorch-fluid` — the library's own documented PyPI distribution (import name is still `flowtorch`). Requires the `FLOWTORCH_DATASETS` env var pointing at the downloaded dataset directory. **The real-data path works** (`FLOWTORCH_DATASETS=/mnt/s/opencode/sage-cfd/data/datasets_29_10_2021/datasets`, run from `phase1/`) and is the basis of the canonical `phase1/results_v2/` results. **float64 is required**: flowTorch's DMD Gram-matrix SVD loses mode orthogonality in float32 once the singular-value ratio drops below ~1e-3 (`phase1/phase1_report.md` §6); `load_cylinder_snapshots(dtype=torch.float64)` is the default.
- Phase 2: `tree-sitter`, `tree-sitter-cpp`, `networkx`, `pyyaml`, `torch` (requirements.txt). **Installed in `dmdsae` conda env** (Python 3.11.15, torch 2.12.1). Run as modules from `phase2/`: `-m extractor.ast_parser`, `-m extractor.operator_graph`, `-m validation.gate1_check`, `-m validation.detect_modification`, `-m rom.composable_operator_rom`.

## 6. Testing & Validation

- **No formal test framework** — validation is `__main__` smoke blocks + `test_synthetic.py` fixture.
- Synthetic fixture: cos/sin spatial-pattern *pairs* at known frequencies (6, 12 Hz) + DC term + noise, shaped exactly like `CylinderSnapshots`. Docstrings document that an earlier single-term `spatial·cos(ωt)` fixture was correctly unfittable by any autonomous linear system — a genuine subtlety, fixed.
- Gate 1 harness: precision/recall of extracted calls vs YAML answer key, per-line candidate pools (handles two calls sharing one line), exit code reflects mismatches/misses.
- Synthetic-modification tests (two tiers): `fvm::Sp(lambda, U)` (known, easy → labeled `source_implicit`, high confidence) and `customForcing::spongeSink(U, spongeCoeff)` (unknown namespace, hard → labeled `UNRECOGNIZED`, routed to review).

## 7. Implementation Status

| Component | Status |
|---|---|
| Phase 1 POD/DMD logic | Verified against synthetic fixture and real data; corrected protocol (`run_v2`) canonical, see `phase1/phase1_report.md` |
| Phase 1 NeuralROM (Arm 1) | **Dropped** on `DMD_POD` (`PATH_FORWARD.md` §0.1); not re-evaluated |
| Phase 1 real-data path | **Working**, float64, `dmdsae` env, flowtorch 1.6.1 (`phase1/results_v2/`) |
| Phase 2 AST extraction | Working on icoFoam (10/10); extended to pimpleFoam v2006 with unqualified-call capture and dispatch expansion |
| fvSchemes fusion | Working |
| Operator graph edges | Working (phiHbyA trace confirmed on icoFoam); pimpleFoam v2006 dispatch expansion verified by `check_pimplefoam_graph.py` |
| Ground truth YAML (icoFoam) | **Reviewed 2026-08-27**, all 10 entries `reviewed:true`, harness passes without warnings |
| Ground truth YAML (pimpleFoam v2006) | **Draft**, 51 entries, all `reviewed:false` — needs human review |
| llm_labeler | Interface/prompt only; `stub_label` raises `NotImplementedError` |
| ComposableOperatorROM | Wiring smoke-tested on random synthetic data only; **retained but not the current Phase 2 test vehicle** — Arm 2/Arm 3 extra blocks are bias-free linear maps of the same span, superseded by the static Step 4 design (`PATH_FORWARD.md` §4) |
| Real CFD training/validation | Not started; static (closed-form) Phase 2 test (`PATH_FORWARD.md` Step 4) is the next planned step |

## 8. Known Issues, Fragilities & Uncertainties

1. **Ground truth circularity risk** — **RESOLVED 2026-08-27, human review confirmed 2026-09-06**: all 10 entries reviewed against OpenFOAM-dev source and Programmer's Guide v2512. Gate 1 runs without unreviewed-entry warnings. Human GT review independently PASSED (2026-09-06) — the "top structural blocker" called out in the 2026-09-06 review is cleared.
2. **`fvi::` namespace — CONFIRMED, labels finalized**: `OpenFOAM-dev/src/finiteVolume/finiteVolume/fvi/` exists on GitHub master, declares `InNamespace Foam::fvi`, header docstrings state operators return `volInternalField` (internal-cell-only, no boundary handling). Confirmed against v11's `icoFoam.C` (still `fvc::`) vs dev's `icoFoam.C` (`fvi::` + `U.internalFieldRef()`). **`fvi::div(phiHbyA)` labeled `flux_divergence`** (not `convection`) — appears in the pressure equation acting on a face-flux field. **`fvm::laplacian`** is the same operator in both cases (page 40, eq 3.14: `∫_V ∇•(Γ∇φ) dV`); single `diffusion` label retained, composition edges capture equation context. All fvi:: `LOW CONFIDENCE` flags removed; `discretization_role` updated to `explicit_internal`.
3. **Documented library bug dependency — CONFIRMED against upstream source** (flowtorch `analysis/dmd.py`): the `dynamics` property's single-matrix branch builds its Vandermonde matrix with legacy `pt.vander` (descending powers), while `.predict()` and even the *list* branch of `dynamics` use `pt.linalg.vander` (ascending) — internally inconsistent three ways; `.reconstruction` inherits the descending convention. The repo's workaround (use `.predict()` only, never `.reconstruction`) is sound. Caveats: (a) verified against upstream master, not the exact installed version (flowtorch is not installed in this environment); (b) the specific experiment magnitudes quoted in the docstring (139% reconstruction error, 1e-6 predict error on an exact linear system) are plausible given the mechanism but were not independently reproduced here. Re-verify if flowtorch is upgraded.
4. **RETRACTED/CORRECTED (2026-09-17)** — was: "DMD blows up on the transient-including window (CONFIRMED 2026-09-05): ... extrapolation error is 6.03e9 ... 39/63 eigenvalues >1 ... category-level limitation of linear DMD." **This does not survive the corrected evaluation protocol.** The 6.03e9 figure was a 400-step rollout-from-t=0.025 combined with float32 precision loss; restarting from the last training snapshot in float64, the same legacy rollout at rank 63 is 7.45%, and the primary (restart-based) forecast at rank 63 is 1.13% (full window) / 0.0013% (post-transient window). See `phase1/phase1_report.md` §5–6 for the full retraction and evidence.
5. **RETRACTED/CORRECTED (2026-09-17)** — was: "Data-efficiency curve fragility ... On the transient window DMD's extrapolation error increases with training size (0.36 → 6.03e9 at n=20→319)." The 6.03e9 endpoint is the same rollout/float32 artifact as item 4. The data-efficiency curve was also computed on the *first* n snapshots (pure startup transient), not the *last* n before the split; recomputed correctly, error is flat-to-slightly-improving from n=50 to n=160 and rises only because n=319 pulls in the transient. See `phase1/phase1_report.md` §4.5, §5.
6. **RETRACTED (2026-09-17)** — was: "Neural ROM long-horizon limitation ... improves to 35.3% extrapolation error." NeuralROM (Arm 1) is **dropped** on the `DMD_POD` branch (`PATH_FORWARD.md` §0.1): its 35.3% figure was worse than the training-mean predictor (29.3%), i.e. not a working baseline. Not re-evaluated. See `phase1/phase1_report.md` §1, §5.
7. **Phase 2 graph details worth noting**:
   - Composition-edge ordering relies on node insertion (source-line) order within each `assigned_to` group — correct for these fixtures but implicit.
   - `build_from_operator_graph` with `collapse_duplicates=True` merges nodes with the same `physical_type`. **FIXED**: unrecognized calls now get `UNKNOWN:{qualified_name}` instead of flat `"UNKNOWN"`, preventing silent collision between different custom namespaces (bug_updates #3, applied 2026-08-27).
   - `fvschemes_parser` regex handles only flat, non-nested blocks (fine for fvSchemes, fragile for general dicts).
   - `detect_modification.diff_extractions` matches multiplicities correctly but is O(n²) via list.remove — irrelevant at this scale.
8. **Resolved questions**: `fvi::div(phiHbyA)` now labeled `flux_divergence` (not `convection`); `fvm::laplacian` velocity-vs-pressure diffusion split **not warranted** — same operator ( Programmer's Guide eq 3.14), composition edges capture equation context; `fvm::Sp` remains `source_implicit` (sufficient for syntax-only pass).
9. **Two historical bugs found and fixed during scaffolding** (fixes present in current code):
   - `_find_enclosing_lhs` used a fixed 4-hop parent walk, silently losing `assigned_to` on terms of a 4-term sum; now walks unbounded through compositional node types.
   - `extract_calls` filtered to `{fvm, fvc, fvi}` at extraction time, dropping custom namespaces before ontology lookup could flag them; filtering now happens only at lookup.
10. **RETRACTED/CORRECTED (2026-09-17)** — was: "`pod_rank_selection.json` stale snapshot-count label — FIXED 2026-09-06." This item is superseded: `pod_baseline.cumulative_energy` was also found to use `s`, not `s²` — the true 99%-energy rank on the full window is **11**, not 63 (`phase1/phase1_report.md` §5, `run_metadata.json`). The corrected protocol (`run_v2`) supersedes both the old rank-selection logic and its label bug; historical `results/` files remain retained for provenance only.
11. **RETRACTED/CORRECTED (2026-09-17)** — was: "DMD fixed-rank sweep on the transient window — confirms rank pollution; blowup confined to rank 63 (2026-09-06) ... sweet spot rank 32: 0.222/0.243 ... blowup only at rank 63 (39 unstable eigenvalues)." **This whole finding was a protocol artifact**, not a real phenomenon: it combined the from-t0 rollout scoring with float32 precision loss and the `s`-vs-`s²` rank mislabeling. Under the corrected protocol (float64, restart-from-last-training-snapshot forecast), linear DMD on the developed regime sits at 1.2–2.3× the POD projection floor across ranks ≥9 with no blowup at any tested rank, and **nonlinear DMD variants are closed as unnecessary for this dataset** (`PATH_FORWARD.md` §4). See `phase1/phase1_report.md` §5–7 for the full retraction, evidence, and the closed decision.

### New findings (2026-09-17)

12. **Solver mismatch found and addressed.** The dataset (`of_cylinder2D_binary`) was produced by **OpenFOAM.com v2006 `pimpleFoam`, laminar** (confirmed from `log.pimpleFoam`), not the OpenFOAM.org/dev `icoFoam` fork Phase 2 originally parsed — a different fork, different namespaces (`fvi::` does not exist in v2006). A new fixture, `phase2/fixtures/pimpleFoam_v2006/`, was pulled from the real v2006 source (gitlab.com/openfoam/core/openfoam, tag `OpenFOAM-v2006`, commit `b45f8f6`) to align the parsed solver with the data (`PATH_FORWARD.md` Step 3, §10 below).
13. **Extractor gaps found and fixed while building the pimpleFoam v2006 fixture**: unqualified (member/free-function) call capture, now opt-in via `include_unqualified`; term sign and side (lhs/rhs) tracking; a case-driven virtual-dispatch expansion of `turbulence->divDevReff(U)` into its concrete `linearViscousStress` terms, resolved via `phase2/extractor/dispatch.py` against the case's `constant/turbulenceProperties` (laminar, model `Stokes` — see fixture's `PROVENANCE.txt` for the full call chain); and a fixed `fvSchemes` key regex. Nested sub-expression calls (e.g. the `fvc::grad(U)` inside the expanded `dev2(T(grad(U)))` term) are marked `nested_in` and excluded via `physical_terms()`, so they are not double-counted as separate equation terms. Verified by `phase2/validation/check_pimplefoam_graph.py`.
14. **Open**: the draft `ground_truth_pimpleFoam_v2006.yaml` (51 entries, all `reviewed: false`) needs human review before any Gate 1 result against it counts (suggested scope: operator-bearing entries — see the file's own header for what's in/out of scope). Two ontology mislabels are pending review on the expanded viscous term: the `fvc::div` of `nuEff*dev2(T(grad U))` is currently labeled `convection_explicit`, and the nested `fvc::grad(U)` inside it is labeled `pressure_gradient_explicit` — both look like mislabels for this term and need a reviewer's correction (`phase2/extractor/ontology.py`). Also open: mapping `fvSchemes` keys from their C++ text form to runtime field names is not implemented, and data-flow edges ignore statement order/branches.
15. **Phase 2 function-class issue (carried forward from item 7, now resolved by design change, not by fixing the ROM)**: Arm 3's extra blocks in the trained `ComposableOperatorROM` are bias-free linear maps — the same function-class span as Arm 2 — so a trained-ROM comparison cannot show a real gap by construction. This is **superseded by the static Step 4 design** (`PATH_FORWARD.md` §0.3, §4): closed-form least-squares fits (A, term-family selection; then B, term-wise projected operators), not gradient-trained blocks.
16. **For the laminar/no-MRF/no-fvOptions case, parsing adds no continuum term beyond textbook Navier–Stokes.** `dev2(T(grad(U)))` = (1/3)ν∇(∇·U) ≈ 0 for divergence-free flow; the other extra terms found by parsing (`ddtCorr`, SIMPLEC-consistent terms) are numerical, not physical. So **Step 4a (the unmodified-solver control) is expected to be a null result by construction** — this is the intended control, not a failure (`PATH_FORWARD.md` §5 Progress Log, 2026-09-17).

## 9. Key Design Assumptions

- DSL function names encode physics → deterministic ontology suffices; LLM only for unresolved calls, constrained to a fixed label set with explicit `UNRECOGNIZED` escape (never silently coerced into the graph).
- Extraction captures *all* qualified calls; namespace narrowing happens only at ontology lookup (so unknowns surface instead of vanishing).
- PISO control structure is a hand-specified skeleton, not learned/inferred — "don't re-discover what's already textbook."
- Single-trajectory dataset means "generalization" = temporal extrapolation only, not cross-Reynolds number — narrower claim, deliberately documented.
- Velocity U (not vorticity) as state variable, to match what Phase 2's operator graph acts on, keeping baseline/arm comparisons meaningful.
- Arm 2 vs Arm 3 must share identical model class, widths, and training so a win isolates "code-parsing helped," not "bigger model."
- **Phase 1 snapshot window includes the transient (2026-09-05)**: `load_cylinder_snapshots()` defaults to the first non-zero time step (t=0.025s) through t=10.0s — 400 snapshots — instead of the flowTorch tutorials' post-transient t>=4.0s window (241). Deliberate harder test case: the transient is where nonlinear, non-periodic structure matters, so validating baselines against the full window before Phase 2 answers whether structured nonlinear arms are worth building. `t_min=4.0` remains an opt-in to reproduce the post-transient-only result. The prior Phase 1 report (`results_ini/phase1_report.md`, 2026-09-01) is post-transient-only and is **not comparable** to the new default. Validation CONFIRMED the decision's premise (2026-09-05): DMD, near-perfect post-transient (0.26%), blows up to 6.03e9 on the transient window (§8.4), while NeuralROM improves — i.e. the window is exactly where nonlinearity matters.

## 10. External Verification of Factual Claims

Factual assertions made by the repo's docstrings/READMEs were independently checked against primary sources (PyPI registry, upstream GitHub repos, OpenFOAM source docs, flowTorch tutorial notebooks). Results:

| Claim (source) | Verdict | Evidence |
|---|---|---|
| PyPI `flowtorch` is an unrelated Meta normalizing-flows package (phase1 README) | **Confirmed** | PyPI JSON API: "Normalizing Flows for PyTorch", Meta Platforms, `facebookincubator/flowtorch` |
| Install via `git+https://github.com/FlowModelingControl/flowtorch.git` (phase1 README) | **Outdated** | Canonical repo is `AndreWeiner/flowtorch` (old org redirects); library's own docs say PyPI distribution is `flowtorch-fluid` |
| flowtorch DMD `.reconstruction`/`.dynamics` Vandermonde bug; use `.predict()` only (`dmd_baseline.py`) | **Confirmed** in mechanism | Upstream `analysis/dmd.py`: `dynamics` single-matrix branch uses legacy `pt.vander` (descending), `.predict()` and the list branch use `pt.linalg.vander` (ascending). Experiment magnitudes (139% / 1e-6) not independently reproduced — no deps installed here |
| `fvi::` namespace real in OpenFOAM-dev; fixture faithful | **Confirmed** | `src/finiteVolume/finiteVolume/fvi/` on master; dev icoFoam.C line-for-line identical to fixture; v11 still uses `fvc::` |
| OpenFOAM-dev foamRun restructuring, classic solvers under `legacy/` (phase2 README) | **Confirmed** | `applications/solvers/` contains only boundaryFoam, chemFoam, foamMultiRun, foamRun, potentialFoam; icoFoam at `applications/legacy/incompressible/icoFoam/` |
| Dataset: Re=100, shedding ~t=1.5s / developed ~4s, crop box [0.1,−1]–[0.75,1], 401 snapshots (400 + t=0) (`data_loading.py`, `run_phase1.py`) | **Confirmed** | flowTorch's own `svd_cylinder.ipynb` / `dmd_cylinder.ipynb`: Re equation =100, "400 snapshots without time '0'", "241 snapshots t≥4.0s", identical mask box and window code. NOTE 2026-09-05: the tutorials' t>=4.0 / 241-snapshot window is no longer the repo default — `load_cylinder_snapshots(t_min=None)` now includes the full transient (400 snapshots), see §9 |
| dt = 0.025s | **Consistent (indirect)** | 400 intervals spanning t=0…10s ⇒ 0.025; tutorials compute dt from `times[1]-times[0]` same as this repo |
| "~13.7k cells" full domain (`phase1_report.md`; `data_loading.py` says "not independently confirmed here") | **Verified 2026-09-06** | Checked on the real dataset: `loader.vertices.shape[0]` = **13,678** cells; ROI `mask_box` [0.1,−1]–[0.75,1] selects **7,190** (`mask.sum()`), data matrix rows = 2·7,190. Matches `phase1_report.md` (13,678 / 7,190). Docstring caveat can be retired |
| pimpleFoam v2006 fixture source provenance (`phase2/fixtures/pimpleFoam_v2006/PROVENANCE.txt`) | **Confirmed** | Fetched from `gitlab.com/openfoam/core/openfoam`, tag `OpenFOAM-v2006` → commit `b45f8f6f587ce22dfe85aa87e97ed72b6afe44b5`; every fixture file's `git hash-object` matches the upstream blob id reported by the GitLab API at that commit. Patch-level check: GitLab compare `OpenFOAM-v2006` → `maintenance-v2006` shows no change to any solver/turbulence-model file used |
| Dataset solver and version is v2006 pimpleFoam, laminar, not icoFoam | **Confirmed** | `log.pimpleFoam` header reports `Build : v2006 OPENFOAM=2006`, solver `pimpleFoam`, `constant/turbulenceProperties: simulationType laminar` — this is what the fixture and ground truth in item above were built to match, superseding the earlier icoFoam-only parsing target |

## 11. Bottom Line

**As of 2026-09-17**, the repository's Phase 1 story has been substantially rewritten: the 2026-09-05/09-06 headline numbers (6.03e9 DMD blowup, ~0.3 "sweet spot" through rank 32, NeuralROM at 35%) were all traced to a flawed evaluation protocol — scoring by a 400-step rollout from t=0.025 instead of a forecast from the last training snapshot, an `s`-vs-`s²` energy convention that mis-selected rank 63 as "99% energy" when the true rank is 11, and float32 precision loss in flowTorch's DMD Gram-matrix SVD. Under the corrected protocol (float64, restart-from-last-training-snapshot, explicit reference predictors), **linear DMD sits essentially at the POD projection floor on the developed regime (1.2–2.3× floor for ranks ≥9), closing the Phase 1 exit criterion: nonlinear DMD variants (HODMD, kernel DMD, EDMD, piDMD, deep Koopman) are not needed for this dataset.** NeuralROM (Arm 1) is dropped on this branch — see `phase1/phase1_report.md` (rewritten 2026-09-17) for the full corrected results and `PATH_FORWARD.md` §0, §4 for the decision record. The still-valid Phase 2 findings from earlier reviews carry forward unchanged: the Vandermonde bug workaround (`.predict()` only), the `fvi::` namespace verification, the icoFoam ground-truth human review (10/10 `reviewed:true`, Gate 1: 100/100/100), and the Meta `flowtorch` PyPI name-clash finding.

A solver mismatch was also found and addressed: the dataset was produced by OpenFOAM.com v2006 `pimpleFoam` (laminar), not the OpenFOAM.org/dev `icoFoam` fork Phase 2 had been parsing. A new fixture (`phase2/fixtures/pimpleFoam_v2006/`) pulled from the verified v2006 source, extended extractor capabilities (unqualified calls, sign/side tracking, case-driven virtual-dispatch expansion of `divDevReff` into its `linearViscousStress` terms via `dispatch.py`), and a draft ground truth (51 entries) bring Phase 2's parsing target in line with the actual data — see §8 items 12–16.

**Next steps** (from `PATH_FORWARD.md`'s plan, in order): (1) human review of `ground_truth_pimpleFoam_v2006.yaml` (51 entries, plus the two pending ontology mislabels on the expanded viscous term); (2) Step 4a — the static, closed-form Phase 2 test (term-family least-squares fit on POD coefficients, A then B design) run as a control on the existing unmodified-solver data, expected to be a null result by construction, since parsing adds no continuum term beyond textbook NS for this laminar case; (3) Step 5 — pre-register the decision rule for "Arm 3 beats Arm 2" before looking at any treatment result; (4) Step 4b — install OpenFOAM locally and run a modified solver with a genuinely out-of-span term (e.g. a non-polynomial drag term), then repeat static test A on that data; the term-wise projected-operator version (B) follows once OpenFOAM mesh operators are available. The trained `ComposableOperatorROM` is retained in the codebase but is not the current Phase 2 test vehicle, because its Arm 2/Arm 3 extra blocks are bias-free linear maps of the same span and cannot show a real gap by construction.
