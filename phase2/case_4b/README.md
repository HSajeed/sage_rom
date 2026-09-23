# Step 4b case kit: `pimpleDragFoam`

Status: **UNTESTED**. Everything in this directory was written and
reviewed but never built or run -- there is no OpenFOAM v2006 install in
this environment. The owner runs `Allrun_4b.sh` (needs a working
OpenFOAM.com v2006 environment sourced, i.e. `wmake`/`runApplication`/
`blockMesh`/`pimpleFoam` on `$PATH` and `$WM_PROJECT_DIR` set) and reports
back the two solver runs so Step 4b's real (not synthetic) result can be
scored.


> **DO NOT RUN AS-IS: the drag term fails the pre-run detectability check (2026-09-23).**
> `python -m static_rom.detectability --amplitude 0.3` (output:
> `detectability_quadratic_drag_cD0.3.json`) shows that on this flow's data manifold
> Φᵀ(|Ũ|Ũ) lies almost entirely in Arm 2's span {1, z, z⊗z}. The out-of-span
> fraction is 7.6e-4 at r=8 and 1.8e-7 at r=23 (full window). The injected
> out-of-span one-step signal is 180–4,600× below the Step 5 threshold
> (3 × Arm 2 validation residual) on the full window, and further below on the
> post-transient one. Reason: the shedding flow lives on a low-dimensional
> near-periodic manifold, so any smooth function of the state is well
> approximated by a quadratic in r ≥ 8 POD coordinates. A negative result
> from this design would be uninformative. The solver scaffold, parser rule
> and `quadratic_drag` plumbing remain valid for other injected terms; the
> choice of term is pending an owner decision.

## What this is

Step 4a (`phase2/results_step4a/`) is a **null control**: on the
unmodified dataset, Arm 3 (parsed UEqn terms) and Arm 2 (textbook
const/lin/quad) get the *same* regressor family span by construction, so
Arm 3 cannot beat Arm 2 -- there is nothing outside {const, lin, quad} in
laminar, no-MRF, no-fvOptions incompressible NS. Step 4b is the
**treatment**: a solver modification that adds a term genuinely outside
that span, so a real (not manufactured) Arm 3 vs Arm 2 gap becomes
possible.

`pimpleDragFoam` is OpenFOAM.com v2006 `pimpleFoam` with one line added to
`UEqn.H`:

```cpp
tmp<fvVectorMatrix> tUEqn
(
    fvm::ddt(U) + fvm::div(phi, U)
  + MRF.DDt(U)
  + turbulence->divDevReff(U)
  + fvm::Sp(cD*mag(U), U)          // <-- added
 ==
    fvOptions(U)
);
```

## Why this is out-of-span

`fvm::Sp(coeff, U)` assembles `coeff*U` into the momentum matrix. With
`coeff = cD*mag(U)` this is `cD*mag(U)*U` -- quadratic in `U`, but through
`mag(U) = sqrt(U_x^2 + U_y^2)`, a **non-polynomial** function of `U`. No
finite-degree polynomial in the POD coefficients `z` (in particular the
Arm 2 quadratic family `z_i*z_j`) can represent `mag(U)*U` exactly, so
this term is genuinely outside the {const, lin, quad} span -- unlike
`fvm::Sp(lambda, U)` with a constant `lambda` (linear, already in `A z`)
or `fvm::div(phi, U)` (quadratic in `U` because `phi` is linear in `U`,
so exactly representable by the quad family after Galerkin projection --
see `static_rom/term_library.py`'s module docstring for that argument in
full). `static_rom/opinf.py`'s `quadratic_drag` extra family is
`Phi^T(|Utilde| Utilde)` -- the SAME non-polynomial functional form,
evaluated pointwise on the reconstructed field and projected -- so Arm 3
can fit this term's contribution essentially exactly while Arm 2 can only
locally approximate it with a polynomial. `static_rom/test_static_rom.py`
`check_drag_synthetic_injection` demonstrates this on a synthetic system
with a known drag coefficient (recovers it to ~13% and beats Arm 2's
validation residual by >25x, safely above the required >3x margin).

## What's verbatim vs reconstructed

Verbatim (copied byte-for-byte from `fixtures/pimpleFoam_v2006/`, itself
fetched from the OpenFOAM.com v2006 tag -- see that directory's
`PROVENANCE.txt`), with the modifications noted:
- `pimpleDragFoam/pimpleDragFoam.C` -- `main()` body is byte-identical to
  `fixtures/pimpleFoam_v2006/pimpleFoam.C`; only the header comment block
  (Application name, this note) differs.
- `pimpleDragFoam/pEqn.H` -- byte-identical to
  `fixtures/pimpleFoam_v2006/pEqn.H` (unaffected by the UEqn change).
- `pimpleDragFoam/UEqn.H`, `pimpleDragFoam/createFields.H` -- copies of
  `fixtures/pimpleFoam_v2006_drag/UEqn.H` and `.../createFields.H` (the
  `+ fvm::Sp(cD*mag(U), U)` addition and the `cD` dimensionedScalar read;
  see `fixtures/pimpleFoam_v2006_drag/PROVENANCE.txt` for the sign-
  convention derivation).

Reconstructed (**not** fetched from upstream, **not** verified against a
real v2006 source tree -- flagged in-file too):
- `pimpleDragFoam/Make/files`, `pimpleDragFoam/Make/options` -- rebuilt
  from the well-documented `applications/solvers/incompressible/
  pimpleFoam` include/library layout, by memory/convention, not from a
  fetched copy. **If `wmake` fails on a missing header or unresolved
  symbol, check these two files first** against
  `$FOAM_SOLVERS/incompressible/pimpleFoam/Make/options` in a real v2006
  install.
- `Allrun_4b.sh` -- written from the dataset's own `Allrun` (`cd
  data/datasets_29_10_2021/datasets/of_cylinder2D_binary && cat Allrun`),
  reusing its mesh (`constant/polyMesh` already built there -- this
  script does NOT re-run `blockMesh`/`snappyHexMesh`/`extrudeMesh`, it
  copies the existing mesh) and its `0.org` + `setExprBoundaryFields`
  inlet-velocity setup verbatim.

## cD derivation

Case parameters (from `phase1/phase1_report.md` and the dataset's
`constant/transportProperties` / `system/snappyHexMeshDict`): Schafer-
Turek 2D-1 benchmark, cylinder diameter `D = 0.1 m` (snappyHexMeshDict
`searchableCylinder` radius `0.05`), mean inlet velocity `U_mean = 1.0
m/s`, `nu = 1e-3` (kinematic, rho=1), giving `Re = U_mean*D/nu = 100`.

`fvm::Sp(cD*mag(U), U)` needs `cD*mag(U)` to have dimensions of
`1/time` (so `coeff*U` has the same `m/s^2` dimensions as the other
UEqn terms, kinematic/rho=1 form). Since `mag(U)` is `m/s`, `cD` must
have dimensions `1/length` -- `[0 -1 0 0 0 0 0]` in OpenFOAM's
`[mass length time ...]` ordering.

Target: the drag term should be a modest perturbation, a few percent of
the convective term, not dominate the dynamics.

```
Convective term scale:  |U . grad(U)| ~ U_mean^2 / D = 1.0^2 / 0.1 = 10 m/s^2
Drag term scale:        cD * mag(U) * U ~ cD * U_mean^2 = cD * 1.0 m/s^2   (numerically, cD in 1/m)
Target: drag / convective in [2%, 5%]
    => cD in [0.02, 0.05] * 10 / 1.0 = [0.2, 0.5]   1/m
Chosen: cD = 0.3 1/m   (drag/convective = 0.3/10 = 3%)
```

`Allrun_4b.sh` sets `CD_VALUE=0.3` by default (overridable via the
`CD_VALUE` environment variable).

## How to build

```sh
# with a working OpenFOAM v2006 environment sourced
cd phase2/case_4b/pimpleDragFoam
wmake
# -> $FOAM_USER_APPBIN/pimpleDragFoam
```

## How to run

```sh
cd phase2/case_4b
./Allrun_4b.sh                                    # uses $FLOWTORCH_DATASETS/of_cylinder2D_binary
# or explicitly:
./Allrun_4b.sh /path/to/of_cylinder2D_binary /path/to/work_dir
```

This builds `pimpleDragFoam`, then assembles two case directories under
`work_4b/` (or the given work dir): `baseline/` (mesh + case config
copied from the dataset case, run with unmodified `pimpleFoam`) and
`drag/` (same mesh + case config, `cD` appended to
`constant/transportProperties`, `controlDict`'s `application` switched to
`pimpleDragFoam`). Both keep the dataset's own `controlDict`
(`writeInterval 0.025`, `endTime 10`, same `deltaT`, `writeFormat
binary`) and ROI/mesh unchanged, so a baseline run should closely
reproduce the existing dataset (a useful sanity check before trusting the
drag run: `baseline/` vs the real dataset should differ only by solver
run-to-run floating point noise, not by the presence/absence of the
`fvm::Sp` line, which the baseline `UEqn.H` doesn't have).

## Expected outputs

Two OpenFOAM cases, each with:
- `constant/`, `system/` -- copied case config (same mesh, `0.025` s
  write interval, `t in [0, 10]`);
- `0/`, `0.025/`, ... -- the usual per-write-interval time directories,
  `U`/`p`/`phi` etc.;
- `log.pimpleFoam` (baseline) / `log.pimpleDragFoam` (drag) -- solver run
  logs, useful for a first eyeball check that `cD` is doing something
  (residuals/Courant number shouldn't blow up; runtime should be similar
  to the baseline's).

## How to score

From `phase2/`, using `/home/sajeed/miniforge3/envs/dmdsae/bin/python`:

```sh
# Treatment: Arm 3 parses the drag fixture's UEqn (gets quadratic_drag),
# scored against the drag-case data.
python -m static_rom.run_step4a \
    --data-dir  work_4b/drag \
    --case-dir  work_4b/drag \
    --source-dir fixtures/pimpleFoam_v2006_drag \
    --out results_step4b_drag

# Control: Arm 3 == Arm 2 on the baseline case data (same null control as
# Step 4a, just re-run on the freshly generated baseline data as a sanity
# check that this case/mesh/run reproduces the null).
python -m static_rom.run_step4a \
    --data-dir  work_4b/baseline \
    --case-dir  work_4b/baseline \
    --source-dir fixtures/pimpleFoam_v2006 \
    --out results_step4b_baseline

# Placebo: Arm 2 gets the SAME quadratic_drag extra capacity as Arm 3,
# without any parse licensing it, scored against the SAME drag-case data
# -- if Arm 2 improves just as much as the real (parsed) Arm 3 run above,
# the gain is "more regressor capacity helps" not "the parser found
# something": read this alongside results_step4b_drag before concluding
# anything.
python -m static_rom.run_step4a \
    --data-dir  work_4b/drag \
    --case-dir  work_4b/baseline \
    --source-dir fixtures/pimpleFoam_v2006 \
    --placebo-drag \
    --out results_step4b_placebo
```

(`--case-dir` only needs to point at a directory `case_activity.py` can
read `constant/turbulenceProperties`/`transportProperties`/`MRFProperties`/
`fvOptions` and `system/fvSchemes`/`fvSolution` from -- `baseline` works
for the placebo run since case activity is identical between the two
cases, only the injected UEqn term differs.)

Apply `DECISION_RULE_step5.md`'s pre-registered thresholds
(`phase2/noise_band.json` from Step 4a) to `results_step4b_drag/grid.csv`
to decide whether Arm 3 beats Arm 2 by more than the Step 4a noise band,
on the primary (discrete) target, at >=3 of the 4 ranks, in the full
window -- exactly as written before any 4b data existed.
