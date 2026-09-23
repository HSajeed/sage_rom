# SAGE-CFD

**Can the structure of a CFD solver, parsed from its source code, give a better reduced-order model than hand-picked textbook physics?**

SAGE-CFD reads the C++ source of an OpenFOAM solver and extracts the terms of the momentum equation that is actually solved. It then uses those terms to choose the regressors of a reduced-order model (ROM), which it compares with a ROM built from textbook Navier–Stokes structure. The test case is the 2D laminar cylinder wake (Re = 100) from the [flowTorch](https://github.com/AndreWeiner/flowtorch) datasets.

| Arm | Model structure |
|---|---|
| **Arm 2**: hand-specified | textbook incompressible NS projected onto POD modes: `dz/dt ≈ c + A z + H(z⊗z)` |
| **Arm 3**: solver-derived | the terms parsed from the solver (`pimpleFoam`, OpenFOAM.com v2006) and the case configuration, mapped to regressor families |

POD and DMD serve as linear reference baselines (Phase 1). The comparison is deliberately **static**: every model is a closed-form, deterministic least-squares fit, with no gradient training and no random seeds.

## Status

| Stage | State |
|---|---|
| Phase 1: POD/DMD baselines, corrected protocol | done |
| Phase 2: extractor + reviewed ground truth (icoFoam, pimpleFoam v2006) | done |
| Phase 2 Step 4a: static null control on the existing dataset | done |
| Phase 2 Step 5: pre-registered decision rule | draft |
| Phase 2 Step 4b: modified-solver treatment run | design open (see below) |

Active development is on the **`DMD_POD`** branch.

## Key results so far

**Phase 1** (`phase1/results_v2/`, float64). Each forecast starts from the last training snapshot and runs the 81-step test window.
- On the developed flow (fit window 4 ≤ t < 8), linear DMD sits at **1.2–2.3× the POD projection floor**, e.g. 0.35% error at rank 11. Nonlinear DMD variants are not needed for this dataset.
- Earlier reports of DMD "blowup" were artifacts of three things: a rollout from t = 0, an `s`-vs-`s²` energy convention, and float32 precision loss.

**Phase 2, Step 4a** (`phase2/results_step4a/`)
- **Null control:** on the unmodified solver, Arm 3 and Arm 2 get identical regressor libraries. They produce bit-identical fits, as they should.
- **Sanity checks:** a λ=0 linear fit reproduces the Phase 1 DMD errors to 5e-10, and the POD floor matches exactly.
- **Transient-including fit window (0 < t < 8):**
  - The quadratic model `c + Az + H(z⊗z)` forecasts at **1.00–1.01× the POD floor** (ranks 8–23).
  - Linear DMD sits at 3.2–4.2× the floor.
  - A library ablation attributes this gain to the quadratic (convective) family.
- **Step 4b design check:** the planned drag injection, `fvm::Sp(cD*mag(U), U)`, turns out to be **undetectable**. On this flow's low-dimensional shedding manifold, |U|U is almost exactly a quadratic in the POD coordinates.
  - `static_rom/detectability.py` now checks any candidate injected term before a simulation is spent on it.
  - The choice of 4b term is open.

## Repository layout

```
phase1/                 POD + DMD baselines and evaluation protocol
  data_loading.py         flowTorch loader, ROI mask, train/test split
  pod_baseline.py         POD (energy from s²), projection floor
  dmd_baseline.py         DMD fit + forecast (flowTorch .predict())
  run_phase1.py           corrected protocol (run_v2) → results_v2/
  test_eval_protocol.py   synthetic-fixture checks
  phase1_report.md        full Phase 1 write-up
phase2/                 parsed-structure vs textbook-structure test
  extractor/              tree-sitter AST → operator ontology → operator graph
  validation/             ground truth (reviewed) + Gate 1 and graph checks
  fixtures/               real OpenFOAM source (icoFoam, pimpleFoam v2006, drag variant)
  static_rom/             Step 4 static test: term library, OpInf fits, runner, detectability
  results_step4a/         Step 4a outputs
  DECISION_RULE_step5.md  pre-registered "Arm 3 beats Arm 2" rule (draft)
  case_4b/                modified-solver kit (untested, do not run as-is)
  rom/                    earlier trained composable ROM (retained, not the test vehicle)
status.md               detailed technical status and findings log
```

Historical results directories (`phase1/results`, `results_ini`, `results_dmdpod`) are kept for provenance only; `results_v2` is canonical.

## Setup

Python 3.11 with float64 PyTorch. The environment used for development:

```bash
pip install flowtorch-fluid                 # NOT "flowtorch" (an unrelated Meta package)
pip install -r phase2/requirements.txt      # tree-sitter, tree-sitter-cpp, networkx, pyyaml, torch
```

The dataset is a separate flowTorch download (not included; `data/` is gitignored). Point flowTorch at it:

```bash
export FLOWTORCH_DATASETS=/path/to/flowtorch/datasets
python -c "from flowtorch import DATASETS; print('of_cylinder2D_binary' in DATASETS)"   # must print True
```

## Running

```bash
# Phase 1 (from phase1/)
python run_phase1.py                 # corrected protocol → results_v2/
python test_eval_protocol.py

# Phase 2 (from phase2/, run as modules)
python -m validation.gate1_check --strict-labels     # extractor vs reviewed ground truth
python -m validation.check_pimplefoam_graph          # pimpleFoam v2006 UEqn terms
python -m static_rom.test_static_rom                 # 28 synthetic checks
python -m static_rom.run_step4a                      # Step 4a grid, null control, noise band (~4 min)
python -m static_rom.detectability --amplitude 0.3   # pre-run check for a 4b injected term
```

## Further reading

- `phase1/phase1_report.md`: Phase 1 results and the corrected evaluation protocol.
- `phase2/README.md`: extractor, ground truth and the static test in detail.
- `status.md`: the complete findings log, including retracted claims and why they were retracted.
