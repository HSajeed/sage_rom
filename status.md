# Technical Report: sage-cfd

*Repository inspection report — updated 2026-08-27 with bug_updates fixes and GT review.*

## 1. Purpose

**SAGE-CFD** is a research scaffold investigating whether ROMs (reduced-order models) for CFD benefit from operator structure *derived from parsing solver source code* rather than hand-specified textbook physics. The core experiment compares three arms on identical snapshot data:

- **Arm 1**: black-box neural ROM (no structure)
- **Arm 2**: composable-operator ROM with hand-specified textbook incompressible-NS structure
- **Arm 3**: same ROM class, structure derived from AST-parsing OpenFOAM solver source

The hypothesis under test: **Arm 3 beats Arm 2**, proving code-parsing adds structural information a textbook specification misses.

## 2. Architecture / Layout

```
sage-cfd/
├── phase1/                  # Conventional baselines (POD/DMD/neural ROM)
│   ├── data_loading.py      # flowTorch cylinder2D dataset loader
│   ├── pod_baseline.py      # POD fit + shared energy rank selection
│   ├── dmd_baseline.py      # DMD wrapper around flowtorch .predict()
│   ├── neural_rom_baseline.py  # encoder/MLP-dynamics/decoder (Arm 1)
│   ├── metrics.py           # relative L2, timing, data-efficiency harness
│   ├── run_phase1.py        # orchestration + comparison table
│   └── test_synthetic.py    # closed-form synthetic fixture w/ known frequencies
└── phase2/
    ├── extractor/
    │   ├── ontology.py          # fixed fvm/fvc/fvi → physics vocabulary
    │   ├── ast_parser.py        # tree-sitter C++ extraction of DSL calls
    │   ├── fvschemes_parser.py  # regex parser for fvSchemes dicts
    │   ├── operator_graph.py    # fuses AST+fvSchemes into networkx DiGraph
    │   └── llm_labeler.py       # STUB: prompt template + schema, no API call
    ├── rom/composable_operator_rom.py  # single ROM class + Arm2/Arm3 builders
    ├── validation/
    │   ├── ground_truth_icofoam.yaml  # REVIEWED 2026-08-27 (all 10 entries reviewed:true)
    │   ├── gate1_check.py             # extractor precision/recall vs GT
    │   └── detect_modification.py     # identity-based diff between sources
    └── fixtures/                # real OpenFOAM-dev icoFoam.C + fvSchemes,
                                 # plus two injected-modification variants
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

- Phase 1: `torch` + flowTorch (CFD library). The PyPI package named `flowtorch` is an **unrelated Meta normalizing-flows project** — *verified against the PyPI registry*: summary "Normalizing Flows for PyTorch", Copyright Meta Platforms, source `facebookincubator/flowtorch`. **Note:** the phase1 README's install command (`git+https://github.com/FlowModelingControl/flowtorch.git`) has been updated to `pip install flowtorch-fluid` — the library's own documented PyPI distribution (import name is still `flowtorch`). Requires the `FLOWTORCH_DATASETS` env var pointing at the downloaded dataset directory. Real-data path untested (flowtorch-fluid not yet installed; Phase 2 deps installed in `dmdsae` conda env).
- Phase 2: `tree-sitter`, `tree-sitter-cpp`, `networkx`, `pyyaml`, `torch` (requirements.txt). **Installed in `dmdsae` conda env** (Python 3.11.15, torch 2.12.1). Run as modules from `phase2/`: `-m extractor.ast_parser`, `-m extractor.operator_graph`, `-m validation.gate1_check`, `-m validation.detect_modification`, `-m rom.composable_operator_rom`.

## 6. Testing & Validation

- **No formal test framework** — validation is `__main__` smoke blocks + `test_synthetic.py` fixture.
- Synthetic fixture: cos/sin spatial-pattern *pairs* at known frequencies (6, 12 Hz) + DC term + noise, shaped exactly like `CylinderSnapshots`. Docstrings document that an earlier single-term `spatial·cos(ωt)` fixture was correctly unfittable by any autonomous linear system — a genuine subtlety, fixed.
- Gate 1 harness: precision/recall of extracted calls vs YAML answer key, per-line candidate pools (handles two calls sharing one line), exit code reflects mismatches/misses.
- Synthetic-modification tests (two tiers): `fvm::Sp(lambda, U)` (known, easy → labeled `source_implicit`, high confidence) and `customForcing::spongeSink(U, spongeCoeff)` (unknown namespace, hard → labeled `UNRECOGNIZED`, routed to review).

## 7. Implementation Status

| Component | Status |
|---|---|
| Phase 1 POD/DMD/NeuralROM logic | Verified against synthetic fixture only |
| Phase 1 real-data path | **Untested** (dataset unavailable here) |
| Phase 2 AST extraction | Working, claimed 10/10 on real OpenFOAM-dev source |
| fvSchemes fusion | Working |
| Operator graph edges | Working (phiHbyA trace confirmed) |
| Ground truth YAML | **Reviewed 2026-08-27**, all 10 entries `reviewed:true`, harness passes without warnings |
| llm_labeler | Interface/prompt only; `stub_label` raises `NotImplementedError` |
| ComposableOperatorROM | Wiring smoke-tested on random synthetic data only |
| Real CFD training/validation | Not started |

## 8. Known Issues, Fragilities & Uncertainties

1. **Ground truth circularity risk** — **RESOLVED 2026-08-27**: all 10 entries reviewed against OpenFOAM-dev source and Programmer's Guide v2512. Gate 1 now runs without unreviewed-entry warnings.
2. **`fvi::` namespace — CONFIRMED, labels finalized**: `OpenFOAM-dev/src/finiteVolume/finiteVolume/fvi/` exists on GitHub master, declares `InNamespace Foam::fvi`, header docstrings state operators return `volInternalField` (internal-cell-only, no boundary handling). Confirmed against v11's `icoFoam.C` (still `fvc::`) vs dev's `icoFoam.C` (`fvi::` + `U.internalFieldRef()`). **`fvi::div(phiHbyA)` labeled `flux_divergence`** (not `convection`) — appears in the pressure equation acting on a face-flux field. **`fvm::laplacian`** is the same operator in both cases (page 40, eq 3.14: `∫_V ∇•(Γ∇φ) dV`); single `diffusion` label retained, composition edges capture equation context. All fvi:: `LOW CONFIDENCE` flags removed; `discretization_role` updated to `explicit_internal`.
3. **Documented library bug dependency — CONFIRMED against upstream source** (flowtorch `analysis/dmd.py`): the `dynamics` property's single-matrix branch builds its Vandermonde matrix with legacy `pt.vander` (descending powers), while `.predict()` and even the *list* branch of `dynamics` use `pt.linalg.vander` (ascending) — internally inconsistent three ways; `.reconstruction` inherits the descending convention. The repo's workaround (use `.predict()` only, never `.reconstruction`) is sound. Caveats: (a) verified against upstream master, not the exact installed version (flowtorch is not installed in this environment); (b) the specific experiment magnitudes quoted in the docstring (139% reconstruction error, 1e-6 predict error on an exact linear system) are plausible given the mechanism but were not independently reproduced here. Re-verify if flowtorch is upgraded.
4. **Data-efficiency curve fragility**: energy-threshold rank selection is unstable at small n; curve is single-seed, non-monotonic; README says to inspect selected ranks per n and average over multiple seeds before trusting it.
5. **Neural ROM long-horizon limitation**: training loss supervises only ≤10-step rollouts; low train loss ≠ accurate full-length rollout (~27% relative rollout error observed despite low training loss). Characteristic of the training regime, not a bug.
6. **Phase 2 graph details worth noting**:
   - Composition-edge ordering relies on node insertion (source-line) order within each `assigned_to` group — correct for these fixtures but implicit.
   - `build_from_operator_graph` with `collapse_duplicates=True` merges nodes with the same `physical_type`. **FIXED**: unrecognized calls now get `UNKNOWN:{qualified_name}` instead of flat `"UNKNOWN"`, preventing silent collision between different custom namespaces (bug_updates #3, applied 2026-08-27).
   - `fvschemes_parser` regex handles only flat, non-nested blocks (fine for fvSchemes, fragile for general dicts).
   - `detect_modification.diff_extractions` matches multiplicities correctly but is O(n²) via list.remove — irrelevant at this scale.
7. **Resolved questions**: `fvi::div(phiHbyA)` now labeled `flux_divergence` (not `convection`); `fvm::laplacian` velocity-vs-pressure diffusion split **not warranted** — same operator ( Programmer's Guide eq 3.14), composition edges capture equation context; `fvm::Sp` remains `source_implicit` (sufficient for syntax-only pass).
8. **Two historical bugs found and fixed during scaffolding** (fixes present in current code):
   - `_find_enclosing_lhs` used a fixed 4-hop parent walk, silently losing `assigned_to` on terms of a 4-term sum; now walks unbounded through compositional node types.
   - `extract_calls` filtered to `{fvm, fvc, fvi}` at extraction time, dropping custom namespaces before ontology lookup could flag them; filtering now happens only at lookup.

## 9. Key Design Assumptions

- DSL function names encode physics → deterministic ontology suffices; LLM only for unresolved calls, constrained to a fixed label set with explicit `UNRECOGNIZED` escape (never silently coerced into the graph).
- Extraction captures *all* qualified calls; namespace narrowing happens only at ontology lookup (so unknowns surface instead of vanishing).
- PISO control structure is a hand-specified skeleton, not learned/inferred — "don't re-discover what's already textbook."
- Single-trajectory dataset means "generalization" = temporal extrapolation only, not cross-Reynolds number — narrower claim, deliberately documented.
- Velocity U (not vorticity) as state variable, to match what Phase 2's operator graph acts on, keeping baseline/arm comparisons meaningful.
- Arm 2 vs Arm 3 must share identical model class, widths, and training so a win isolates "code-parsing helped," not "bigger model."

## 10. External Verification of Factual Claims

Factual assertions made by the repo's docstrings/READMEs were independently checked against primary sources (PyPI registry, upstream GitHub repos, OpenFOAM source docs, flowTorch tutorial notebooks). Results:

| Claim (source) | Verdict | Evidence |
|---|---|---|
| PyPI `flowtorch` is an unrelated Meta normalizing-flows package (phase1 README) | **Confirmed** | PyPI JSON API: "Normalizing Flows for PyTorch", Meta Platforms, `facebookincubator/flowtorch` |
| Install via `git+https://github.com/FlowModelingControl/flowtorch.git` (phase1 README) | **Outdated** | Canonical repo is `AndreWeiner/flowtorch` (old org redirects); library's own docs say PyPI distribution is `flowtorch-fluid` |
| flowtorch DMD `.reconstruction`/`.dynamics` Vandermonde bug; use `.predict()` only (`dmd_baseline.py`) | **Confirmed** in mechanism | Upstream `analysis/dmd.py`: `dynamics` single-matrix branch uses legacy `pt.vander` (descending), `.predict()` and the list branch use `pt.linalg.vander` (ascending). Experiment magnitudes (139% / 1e-6) not independently reproduced — no deps installed here |
| `fvi::` namespace real in OpenFOAM-dev; fixture faithful | **Confirmed** | `src/finiteVolume/finiteVolume/fvi/` on master; dev icoFoam.C line-for-line identical to fixture; v11 still uses `fvc::` |
| OpenFOAM-dev foamRun restructuring, classic solvers under `legacy/` (phase2 README) | **Confirmed** | `applications/solvers/` contains only boundaryFoam, chemFoam, foamMultiRun, foamRun, potentialFoam; icoFoam at `applications/legacy/incompressible/icoFoam/` |
| Dataset: Re=100, shedding ~t=1.5s / developed ~4s, t≥4.0 crop window [0.1,−1]–[0.75,1], 401 snapshots (400 + t=0), 241 post-transient (`data_loading.py`, `run_phase1.py`) | **Confirmed** | flowTorch's own `svd_cylinder.ipynb` / `dmd_cylinder.ipynb`: Re equation =100, "400 snapshots without time '0'", "241 snapshots t≥4.0s", identical mask box and window code |
| dt = 0.025s | **Consistent (indirect)** | 400 intervals spanning t=0…10s ⇒ 0.025; tutorials compute dt from `times[1]-times[0]` same as this repo |
| "~13.7k cells" (`data_loading.py` docstring) | **Unverified** | Not stated in any flowTorch tutorial text checked; confirm with `mask.sum()` / mesh inspection on real data |

## 11. Bottom Line

The repository is unusually honest about its own limits — nearly every caveat above is self-documented in docstrings and READMEs. Independent verification found its specific factual claims to be accurate in mechanism and substance. **As of 2026-08-27**: all 6 bug_updates fixes applied, ground truth fully reviewed (10/10 entries reviewed:true), regression suite passes (Gate 1: 100/100/100, detect_modification: both tiers, ROM smoke test: Arm 3 = 7 blocks, all gradients flow). Phase 2 deps installed in `dmdsae` conda env. The remaining gap to the project's goal is: (1) get flowTorch + dataset working for Phase 1 real-data validation, (2) set up OpenFOAM locally for real snapshot generation, (3) multi-seed training and comparison of Arms 1/2/3. Everything upstream of those steps is scaffolded, tested, and internally consistent.
