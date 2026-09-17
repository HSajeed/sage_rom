# SAGE-CFD -- Phase 2 scaffold

Solver source -> AST-extracted operator calls -> fused with fvSchemes ->
operator graph -> composable-operator ROM (shared architecture for the
hand-specified and solver-derived arms).

Fixtures are **real, current OpenFOAM-dev source**, pulled 2026-08-24 from
`github.com/OpenFOAM/OpenFOAM-dev` (`applications/legacy/incompressible/icoFoam/`),
not reconstructed from memory. Worth knowing: this repo has restructured
around a modular `foamRun` architecture -- classic standalone solvers like
`icoFoam` now live under `applications/legacy/`, and current `icoFoam.C`
uses an `fvi::` namespace (`fvi::grad`, `fvi::div`) that isn't part of the
classic OpenFOAM teaching material. That's a live, small-scale instance of
the exact argument for this whole pipeline: parsing the real source catches
things that memorized/textbook knowledge of the DSL would miss.

## Layout

```
extractor/
  ontology.py            fixed physical-operator vocabulary (fvm/fvc/fvi -> physics)
  ast_parser.py           tree-sitter extraction of fvm::/fvc::/fvi:: calls
  fvschemes_parser.py     parses the case's fvSchemes dict (discretization scheme per operator)
  operator_graph.py       fuses both sources into a networkx graph + PISO skeleton
  llm_labeler.py           STUB -- narrow-scope LLM pass for calls the ontology can't resolve

validation/
  ground_truth_icofoam.yaml            REVIEWED 2026-08-27 -- all 10 entries reviewed:true
  ground_truth_pimpleFoam_v2006.yaml   DRAFT answer key, 51 entries, NOT YET REVIEWED
  gate1_check.py                       precision/recall of the extractor against ground truth
  check_pimplefoam_graph.py            checks pimpleFoam v2006 UEqn assembly (dispatch expansion, sign/side)

rom/
  composable_operator_rom.py   shared ROM class; Arm 2 (hand-specified) and
                                 Arm 3 (solver-derived) builders. Retained but not
                                 the current Phase 2 test vehicle -- see "Next steps" below.

fixtures/
  icoFoam.C, fvSchemes           real OpenFOAM-dev source + case config
  pimpleFoam_v2006/              real OpenFOAM.com v2006 pimpleFoam source + the
                                   dataset's own fvSchemes/turbulenceProperties --
                                   see "pimpleFoam v2006" section below
```

## Running it

```
pip install -r requirements.txt

# Extraction, standalone
python -m extractor.ast_parser fixtures/icoFoam.C

# Fused operator graph (AST + fvSchemes + PISO skeleton)
python -m extractor.operator_graph fixtures/icoFoam.C fixtures/fvSchemes

# Gate 1: extractor vs ground truth
python -m validation.gate1_check

# ROM wiring smoke test (synthetic data, NOT CFD validation)
python -m rom.composable_operator_rom
```

## What's real vs. scaffolded

**Real and working, on real source:**
- AST extraction (10/10 DSL calls correctly identified on `icoFoam.C`,
  including two calls sharing one source line)
- `fvSchemes` fusion (scheme lookup with OpenFOAM's own default-fallback behavior)
- Composition and data-flow edges built automatically from the AST (e.g.
  the extractor correctly traced `phiHbyA` from `fvc::flux`/`fvc::interpolate`/
  `fvc::ddtCorr` through to its use in the pressure equation, with no
  hand-coded rule for that specific variable name)
- Gate 1 harness (already caught one of its own bugs during scaffolding --
  a same-line-collision matching error -- which is itself a small proof
  that having the harness is worth it)
- `ComposableOperatorROM` -- both arms build, run, and receive gradients
  through every block

`ground_truth_icofoam.yaml` was **reviewed 2026-08-27** (all 10 entries
`reviewed: true`); Gate 1 runs against it without unreviewed-entry warnings.
The already-visible Arm 2 vs. Arm 3 gap on icoFoam is **3 blocks vs. 7**, with
Arm 3 picking up PISO-specific bookkeeping terms like `flux_reconstruction`
and `interpolation` -- still a structural observation, not an accuracy
result (see "Next steps" for why this isn't yet the test vehicle).

**Explicitly NOT real yet -- do not mistake these for results:**
- `ground_truth_pimpleFoam_v2006.yaml` (the pimpleFoam v2006 ground truth,
  see below) is still a draft I (Claude) wrote while scaffolding this, not
  independent ground truth. Every entry says `reviewed: false` and
  `gate1_check.py` warns loudly about this every run. A Gate 1 "pass"
  right now proves the harness works, not that the labels are correct --
  review each entry against your own OpenFOAM knowledge before it counts.
- The `fvi::` entries (icoFoam fixture only) are flagged low-confidence in
  `ontology.py` -- worth checking OpenFOAM-dev's own docs/changelog for what
  that namespace actually means before trusting "pressure_gradient" /
  "convection" as their labels. On the pimpleFoam v2006 fixture, two more
  ontology labels are pending review: the `fvc::div` of the expanded
  `nuEff*dev2(T(grad(U)))` viscous term is currently labeled
  `convection_explicit`, and the nested `fvc::grad(U)` inside it is labeled
  `pressure_gradient_explicit` -- both look wrong for this term and need a
  reviewer's correction.
- `llm_labeler.py` is an interface + prompt template, not a live call --
  no API key configured in this environment. `stub_label()` raises
  `NotImplementedError` on purpose.
- The ROM smoke test uses random synthetic data. It proves the
  architecture is wired correctly (forward pass, loss, gradients reach
  every block) and nothing more -- there is no CFD training data in this
  scaffold yet. It is also, by construction, not able to show a real
  Arm 2 vs. Arm 3 gap -- see "Next steps" below.

## Synthetic-modification tests (step 2, done)

Two injected fixtures, both adding a sink/damping term to `UEqn` that a
"textbook incompressible NS" hand-specification (Arm 2) would never
anticipate:

- `fixtures/icoFoam_modified.C` -- **easy case**: `fvm::Sp(lambda, U)`, a
  real, idiomatic OpenFOAM mechanism, already a known ontology entry.
  Detected cleanly, labeled `source_implicit`, confidence `high`.
- `fixtures/icoFoam_modified_unknown.C` -- **hard case**: the same idea
  implemented as hand-rolled custom code, `customForcing::spongeSink(...)`,
  a namespace the ontology has never seen. Detected, correctly labeled
  `UNRECOGNIZED`, correctly routed to "needs LLM/human review" rather than
  silently dropped or silently mislabeled.

Check either with `python -m validation.detect_modification fixtures/icoFoam.C
fixtures/icoFoam_modified.C` (swap in `_unknown` for the hard case) --
matches by call *identity* (namespace, function, arguments), not line
number, so it's immune to line-shift noise from unrelated edits.

Confirmed end to end: the injected term propagates all the way to
`rom/composable_operator_rom.py` -- Arm 3 built from the modified graph
picks up a `source_implicit` block that Arm 2 (built blind, by
construction) structurally cannot have. That's Gate 2's precondition: the
two arms now differ in a way attributable to code-parsing, not to an
architecture difference.

**Two real bugs this testing caught, both fixed, both worth knowing
about if you extend this:**
1. `_find_enclosing_lhs` in `ast_parser.py` used a fixed 4-hop parent walk
   to find which equation a term belongs to. A 4-term sum needs 5 hops --
   the original 3-term `UEqn` was fine, but adding one term silently lost
   `assigned_to` on two of the four terms. Fixed by walking up through
   compositional node types with no hard limit instead of a magic number.
2. `extract_calls` defaulted to filtering to `{fvm, fvc, fvi}` at the
   *extraction* stage everywhere it was called. That's backwards: it means
   a genuinely custom namespace was dropped before ever reaching the
   ontology lookup that's supposed to flag it as unrecognized -- the
   `customForcing::spongeSink` hard-case test initially came back as "0
   added" (silently missed) until this was fixed. Namespace filtering now
   only happens at the ontology-lookup stage; extraction captures every
   qualified call.

Neither bug affected the headline Gate 1 number reported earlier (both
happened to not matter for the unmodified, 3-term, all-known-namespace
case) -- which is itself worth noting: a scaffold "passing" its first test
doesn't mean it's correct, only that the first test wasn't hard enough to
expose what was wrong.

## pimpleFoam v2006 (the dataset's actual solver)

The Phase 1 cylinder2D dataset was produced by **OpenFOAM.com v2006
`pimpleFoam`, laminar** (confirmed from the dataset's own `log.pimpleFoam`
header), not the OpenFOAM.org/dev `icoFoam` fork the original fixture
above was pulled from -- a different fork with different namespaces
(`fvi::` doesn't exist in v2006). `fixtures/pimpleFoam_v2006/` brings the
parsed solver in line with the data it's meant to explain.

**Provenance.** `fixtures/pimpleFoam_v2006/PROVENANCE.txt` documents the
exact source: `gitlab.com/openfoam/core/openfoam`, tag `OpenFOAM-v2006`,
commit `b45f8f6f587ce22dfe85aa87e97ed72b6afe44b5`. Every fixture file's
`git hash-object` is checked there against the upstream blob id reported
by the GitLab API at that commit, and a compare against
`maintenance-v2006` confirms no patch-level change to any file used. The
file also documents the case's `fvSchemes`/`fvSolution` settings and the
laminar viscous-term call chain (see "dispatch.py" below).

**Running it:**

```
python -m validation.check_pimplefoam_graph

python -m validation.gate1_check \
    --source fixtures/pimpleFoam_v2006/UEqn.H \
    --ground-truth validation/ground_truth_pimpleFoam_v2006.yaml \
    --include-unqualified
```

(`gate1_check.py` scores only the ground-truth entries whose
`source_file` matches `--source`, since this is a multi-file fixture --
run it once per file, e.g. swap in `pEqn.H` or `linearViscousStress.C`.
`--include-unqualified` is needed because the ground truth includes
member/free-function calls like `turbulence->divDevReff(U)`, which the
extractor only captures when unqualified-call capture is turned on.)

**`dispatch.py`.** `UEqn.H` calls `turbulence->divDevReff(U)`, a virtual
call whose concrete implementation depends on the case's runtime-selected
turbulence model, not on anything visible at the call site. `dispatch.py`
is a small, hand-curated expansion table (not a general C++ call-graph
resolver) that maps this call, given the case's
`constant/turbulenceProperties`, to its actual body: for this case
(`simulationType laminar`, no `laminar{}` subdict) it assumes the
OpenFOAM-documented default laminar model, **Stokes** -- an assumption
flagged in the code as not independently verified against the
runtime-selection source, only against `Stokes.H`'s own description. Under
that assumption, the call expands to the two `linearViscousStress::divDevRhoReff`
terms (`fvm::laplacian(nuEff, U)` and `fvc::div(nuEff*dev2(T(fvc::grad(U))))`),
with signs and equation side carried through from the outer call.

**`physical_terms()`.** The operator graph (`operator_graph.py`) marks
sub-expression calls that are nested inside another term's arguments (e.g.
the `fvc::grad(U)` inside the expanded `dev2(T(grad(U)))` term) as
`nested_in`; `physical_terms()` is the helper that walks the graph and
excludes anything so marked, so nested calls aren't double-counted as
separate equation terms alongside the term they're part of.

**Ground truth status.** `validation/ground_truth_pimpleFoam_v2006.yaml`
is a **draft**: 51 entries, all `reviewed: false`. It still needs human
review before any Gate 1 result against it counts (see "Explicitly NOT
real yet" above for the two pending ontology mislabels found while
scaffolding it).

## Next steps

The remaining work is tracked in `PATH_FORWARD.md` at the repo root (a
local, gitignored file -- not in git, so it isn't linked here as a
reference, but is the authoritative running plan). In order:

1. **Human review of `ground_truth_pimpleFoam_v2006.yaml`** (51 entries),
   plus the two ontology mislabels flagged above on the expanded viscous
   term, and the still-open icoFoam ontology questions (velocity- vs.
   pressure-diffusion distinction; what `fvi::` actually means; whether
   `fvm::Sp` deserves a more specific `physical_type`).
2. **Step 4a -- static control.** Phase 2 is being redesigned as a
   *static*, closed-form test: least-squares fits on POD coefficients
   (term-family selection, then term-wise projected operators), not
   gradient-trained blocks. Step 4a runs this on the existing,
   *unmodified*-solver dataset as a control -- expected to be a null
   result by construction, since for this laminar/no-MRF/no-fvOptions case
   parsing adds no continuum term beyond textbook Navier-Stokes (the
   `dev2` term is ≈0 for divergence-free flow; the rest is numerical, not
   physical).
3. **Step 4b -- treatment**, requiring a local OpenFOAM install and a
   modified-solver run with a genuinely out-of-span term (e.g. a
   non-polynomial drag term -- `fvm::Sp` alone is linear and already
   representable by the textbook arm, so it can't show a gain), then the
   same static test (B) on that data.

**Note on `ComposableOperatorROM`:** the trained ROM (`rom/`) is retained
in the codebase but is **not** the current Phase 2 test vehicle. Its
Arm 2/Arm 3 extra blocks are bias-free `nn.Linear` maps, so they share the
same function-class span -- a sum of linear blocks is one linear map --
and a trained comparison between the arms cannot show a real gap by
construction. The static design above replaces it for now.
