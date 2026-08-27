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
  ground_truth_icofoam.yaml   DRAFT answer key -- NOT YET REVIEWED, see below
  gate1_check.py                 precision/recall of the extractor against ground truth

rom/
  composable_operator_rom.py   shared ROM class; Arm 2 (hand-specified) and
                                 Arm 3 (solver-derived) builders

fixtures/
  icoFoam.C, fvSchemes           real OpenFOAM-dev source + case config
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

**Explicitly NOT real yet -- do not mistake these for results:**
- `ground_truth_icofoam.yaml` is a draft **I (Claude) wrote while scaffolding
  this**, not independent ground truth. Every entry says `reviewed: false`
  and `gate1_check.py` warns loudly about this every run. A Gate 1 "pass"
  right now proves the harness works, not that the labels are correct --
  review each entry against your own OpenFOAM knowledge before it counts.
- The `fvi::` entries are flagged low-confidence in `ontology.py` for the
  same reason -- worth checking OpenFOAM-dev's own docs/changelog for what
  that namespace actually means before trusting "pressure_gradient" /
  "convection" as their labels.
- `llm_labeler.py` is an interface + prompt template, not a live call --
  no API key configured in this environment. `stub_label()` raises
  `NotImplementedError` on purpose.
- The ROM smoke test uses random synthetic data. It proves the
  architecture is wired correctly (forward pass, loss, gradients reach
  every block) and nothing more -- there is no CFD training data in this
  scaffold yet.
- The already-visible Arm 2 vs. Arm 3 gap (3 blocks vs. 6, with Arm 3
  picking up PISO-specific bookkeeping terms like `flux_reconstruction`
  and `interpolation`) is a structural observation, not an accuracy
  result. Whether those extra blocks help or just add noise is exactly
  what the real experiment needs to answer.

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

## Next steps, in order

1. **Review `ground_truth_icofoam.yaml` by hand.** Still the actual next
   action -- nothing above changes that. Pay particular attention to the
   two open ontology questions already flagged in the file (velocity- vs.
   pressure-diffusion distinction; what `fvi::` actually means), plus a
   third one raised by this session: should `fvm::Sp` get a more specific
   physical_type than the generic `source_implicit`, or is "it's a source
   term, and what kind requires more context" the honest, correct level of
   claim for a syntax-only pass to make?
2. ~~Design and inject the synthetic modification~~ -- done above, in two
   difficulty tiers.
3. **Generate real snapshot data** from the unmodified and modified
   solver (Foam-Agent's case-execution tooling is worth reusing here
   rather than rebuilding).
4. **Train Arm 1 / Arm 2 / Arm 3** on the same snapshots, multiple seeds,
   and check specifically whether Arm 3 beats Arm 2 -- that gap, not
   either arm's absolute accuracy, is the answer to whether Phase 2's
   core premise holds.
