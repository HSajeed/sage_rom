# Labeling notes: pimpleFoam v2006 ground truth

Labels proposed on 2026-09-17 for `ground_truth_pimpleFoam_v2006.yaml`. All 51 entries still have `reviewed: false`.

**Reference.** OpenFOAM (ESI/openfoam.com) Programmer's Guide v2512 (`Openfoam_doc.pdf`). Printed page P-n is PDF index n-1.

**Version gap.** The guide is v2512. The solver and the dataset are v2006.

**Other sources.** Where the guide is silent, labels come from the v2006 source: the solver and turbulence-model fixtures, plus the library files checked in verbatim under `fixtures/pimpleFoam_v2006/upstream/` (tag `OpenFOAM-v2006`, commit b45f8f6f). Those are TensorI.H, fvcDdt.C, fvcDiv.C, fvcFlux.C, fvcMeshPhi.C, EulerDdtScheme.C, ddtScheme.C/.H, ddtSchemeBase.C, MRFZoneList.C, MRFZone.C, fvOptionListTemplates.C, fvMatrix.C, constrainHbyA.C and constrainPressure.C. Their git blob ids are listed in `fixtures/pimpleFoam_v2006/PROVENANCE.txt`, and each was verified with `git hash-object`. Line numbers below refer to those checked-in copies.

**Counts.** 51 entries:
- `operator`: 24. Of these, 19 are `term_kind: term`, 2 are `forwarding` and 3 are `argument` (nested `fvc::grad(U)`).
- `operator_argument`: 3 (nuEff, dev2, T; all `term_kind: argument`).
- `bookkeeping`: 24.

Gate 1 recall is unchanged: 100% for each of the 4 files, and icoFoam is 100/100/100.

## Labeling policy (owner decision, 2026-09-17)

The YAML carries `physical_type_policy_version: "2026-09-17"`.

**Rule 1 (verbatim): physical_type = operator + operand (what the term IS in context, per guide §3.4.2 vs §3.4.5).**
- `fvc::grad(p)` → `pressure_gradient`
- `fvc::grad(U)` → `velocity_gradient`
- generic other → `gradient`
- `fvc::div(phiHbyA)` (face flux) → `flux_divergence`
- `fvc::div(phi,U)`-style two-arg → `convection`
- `fvc::div(nuEff*dev2(T(grad U)))` → `viscous_stress_divergence_explicit`
- `fvc::snGrad(p)` → `pressure_gradient_face_normal` (otherwise `surface_normal_gradient`)

Explicit/implicit lives in `discretization_role`, never in `physical_type`.

- **Known exception.** `fvm::laplacian` stays `diffusion` for every operand, including `laplacian(rAtU(), p)`. This keeps consistency with the reviewed icoFoam key, whose labels are frozen. A strict operand-aware reading would call the pressure Laplacian something like `pressure_laplacian`. Flagged for the owner.
- **Rule 2.** `nuEff`, `dev2` and `T` → `scope: operator_argument`. They stay in the key for Gate 1 call-identity matching but are never ROM terms. Their roles stay `coefficient` / `algebraic`.
- **Rule 3.** `divDevReff` stays expanded.
  - UEqn.H:9 `divDevReff` and ITM.C:117 `divDevRhoReff` are `term_kind: forwarding`.
  - LVS.C:103 `fvm::laplacian(nuEff,U)` and LVS.C:102 `fvc::div(... dev2 ...)` are the real terms (`term_kind: term`).
  - Every operator-scope entry has `term_kind` term | forwarding. Operator_argument entries have `term_kind: argument`.
- **Labeler's application of Rule 3 (please confirm).** An `fvc::` call nested inside another operator's argument keeps `scope: operator` but gets `term_kind: argument`. This applies to `fvc::grad(U)` at LVS.C:102 and :118 (inside `fvc::div(...)`) and at LVS.C:87 (inside `dev(twoSymm(...))`). It matches the graph's `nested_in` handling, and `check_pimplefoam_graph` already asserts the nested `grad(U)` is not a physical term. Nesting inside a bookkeeping wrapper (`solve`, `zeroFilter`, `constrainHbyA`) does not demote a term.
- **Rule 4.** `fvOptions(U)` → `user_source`, role `implicit_or_explicit`. Whether the function name is `fvOptions` or `operator()` is still open.

## What the guide covers and what it does not

**Covered:**
- §1.1.3 p.12: how an equation maps to code (`ddt + div - laplacian == -grad(p)`).
- §1.2.4 pp.17-19: the pisoFoam.C header. It gives the momentum equation with `−∇·R` and says MRF and fvOptions exist as sub-models.
- Ch.3:
  - Operators: eq 3.2-3.11.
  - fvm vs fvc: p.38.
  - Table 3.2: p.39.
  - Laplacian: eq 3.14-3.15.
  - Convection: eq 3.16-3.19.
  - ddt: eq 3.20-3.22.
  - Divergence: eq 3.24, which it explicitly separates from convection.
  - Gradient: eq 3.25.
  - snGrad: eq 3.28.
  - Sources: eq 3.29.
  - surfaceIntegrate, surfaceSum, average and faceInterpolate: §3.4.10.
  - Temporal discretisation: eq 3.30-3.36.
- Appendix A: `dev` (eq A.32) and transpose (eq 2.2).

**Not covered:**
- PISO, PIMPLE and SIMPLEC as algorithms. pisoFoam is only a compile example.
- `fvc::flux`, `fvc::interpolate` (only `faceInterpolate()` is named), `fvc::ddtCorr`, `fvc::makeRelative`, `fvc::makeAbsolute` and `fvc::correctUf`.
- `fvMatrix::A/H/H1/flux`, `relax`, `setReference`, `constrainHbyA`, `constrainPressure` and `adjustPhi`.
- `dev2` and `twoSymm`.
- The `divDevReff` turbulence-model interface, fvOptions and MRF internals.
- The `fvi::` namespace. It appears nowhere in the guide.
- The "Reconstruct" heading in §3.4.10 has no body.

## Operator entries

Abbreviations: G = Programmer's Guide v2512. LVS = linearViscousStress.C. ITM = IncompressibleTurbulenceModel.C.

| file:line | call | scope / term_kind | physical_type | role | doc reference | conf. | rationale |
|---|---|---|---|---|---|---|---|
| UEqn.H:7 | `fvm::ddt(U)` | operator / term | time_derivative | implicit | G §3.4.3 p.41 eq 3.20-3.21; Tab 3.2 p.39 | high | Euler time derivative, exactly as the guide describes it |
| UEqn.H:7 | `fvm::div(phi,U)` | operator / term | convection | implicit | G §3.4.2 p.40 eq 3.16; Tab 3.2 | high | Written `div(psi, phi)` with a face flux: this is the guide's convection term |
| UEqn.H:8 | `MRF.DDt(U)` | operator / term | mrf_coriolis_source | explicit | not in G; fixtures/pimpleFoam_v2006/upstream/MRFZoneList.C:167-194, fixtures/pimpleFoam_v2006/upstream/MRFZone.C:311-333 | medium | Returns a volVectorField of Ω×U, which enters the equation as an explicit source (G §3.4.9) |
| UEqn.H:9 | `turbulence->divDevReff(U)` | operator / **forwarding** | viscous_stress_divergence | implicit+explicit | G §1.2.4 p.18 (`−∇·R`), §1.1.3 p.12; dispatch: ITM.C:117 → LVS.C:102-103 | medium | Forwarding node (Rule 3) for the whole deviatoric viscous term −∇·[ν(∇U+(∇U)ᵀ−⅔(∇·U)I)]. The real terms are LVS.C:102-103 |
| UEqn.H:11 | `fvOptions(U)` | operator / term | user_source *(was source_explicit_user)* | implicit_or_explicit | G §3.4.9 p.43 (general sources); fixtures/pimpleFoam_v2006/upstream/fvOptionListTemplates.C:76-151 | medium | Returns an fvMatrix built with `addSup`, so it is not necessarily explicit |
| UEqn.H:21 | `fvc::grad(p)` | operator / term | pressure_gradient *(was pressure_gradient_explicit)* | explicit | G §1.1.3 p.12; §3.4.6 p.42 eq 3.25; Tab 3.2 | high | This is the guide's own `−fvc::grad(p)` ↔ −∇p. Label matches icoFoam |
| pEqn.H:3 | `fvc::flux(HbyA)` | operator / term | flux_reconstruction | explicit | partial: G §3.4.2 eq 3.16 defines F=S_f·U_f; fixtures/pimpleFoam_v2006/upstream/fvcFlux.C:33-43 | high | S_f·(HbyA)_f. Label kept from icoFoam (see the naming caveat below) |
| pEqn.H:7 | `fvc::interpolate(rAU)` | operator / term | interpolation | explicit | G §3.4.10 p.44 (faceInterpolate) + eq 3.17 | high | Cell-to-face interpolation of 1/A. Version-sensitive: v2006 picks the scheme at run time |
| pEqn.H:7 | `fvc::ddtCorr(U,phi,Uf)` | operator / term | temporal_flux_correction | explicit | not in G; fixtures/pimpleFoam_v2006/upstream/fvcDdt.C:208-228, fixtures/pimpleFoam_v2006/upstream/EulerDdtScheme.C:523-550, fixtures/pimpleFoam_v2006/upstream/ddtScheme.C:143-213 | medium | c_f(φᵒ−S_f·U_fᵒ)/Δt: the transient Rhie-Chow-type flux correction |
| pEqn.H:11 | `fvc::interpolate(rAU)` | operator / term | interpolation | explicit | as pEqn.H:7 | high | Dead else-branch. Suspect dimensions, probably an upstream bug |
| pEqn.H:29 | `fvc::interpolate(rAtU()-rAU)` | operator / term | interpolation | explicit | as pEqn.H:7; SIMPLEC not in G | high | Face value of the SIMPLEC coefficient difference |
| pEqn.H:29 | `fvc::snGrad(p)` | operator / term | pressure_gradient_face_normal *(was surface_normal_gradient)* | explicit | G §3.4.6 p.42 eq 3.28; Tab 3.2 | medium | Face-normal ∂p/∂n in the SIMPLEC flux term. Labeled by what it is in context |
| pEqn.H:30 | `fvc::grad(p)` | operator / term | pressure_gradient *(was …_explicit)* | explicit | G §3.4.6 eq 3.25; Tab 3.2 | high | SIMPLEC correction `HbyA -= (rAU−rAtU)∇p` |
| pEqn.H:46 | `fvm::laplacian(rAtU(),p)` | operator / term | diffusion | implicit | G §3.4.1 p.40 eq 3.14-3.15; Tab 3.2 | high | Same operator as eq 3.14 with Γ=rAtU. Label matches icoFoam. In context it is the pressure-Poisson operator (the Rule 1 exception) |
| pEqn.H:46 | `fvc::div(phiHbyA)` | operator / term | flux_divergence | explicit | G §3.4.5 p.41 eq 3.24 (surface field); Tab 3.2; §3.4.10 | high | Divergence of a face flux (surfaceIntegrate). The guide says this is not convection |
| pEqn.H:64 | `fvc::grad(p)` | operator / term | pressure_gradient *(was …_explicit)* | explicit | G §1.1.3; §3.4.6 eq 3.25 | high | Momentum corrector `U = HbyA − rAtU∇p` |
| ITM.C:117 | `divDevRhoReff(U)` | operator / **forwarding** | viscous_stress_divergence | implicit+explicit | not in G (dispatch); ITM.C:112-118 | medium | Forwarding node (Rule 3). It must not be counted as a second viscous term |
| LVS.C:87 | `fvc::grad(this->U_)` | operator / argument (nested) | velocity_gradient | explicit | G §3.4.6 eq 3.25; §3.1.1 eq 3.3 | high | Inside devRhoReff (stress output). Not on the momentum path |
| LVS.C:102 | `fvc::div(nuEff*dev2(T(grad U)))` | operator / term | viscous_stress_divergence_explicit | explicit | G §3.4.5 p.41 eq 3.24; §3.1.2 eq 3.5; dev2 not in G | medium | ∇·[ν((∇U)ᵀ−⅔(∇·U)I)]. Equals ⅓ν∇(∇·U) when ν is constant: zero in the continuum, small but nonzero in the discrete solver |
| LVS.C:102 | `fvc::grad(U)` | operator / argument (nested) | velocity_gradient | explicit | G §3.4.6 eq 3.25; §3.1.1 eq 3.3 | high | Nested argument of the explicit viscous term |
| LVS.C:102 | `this->nuEff()` | **operator_argument** / argument | effective_viscosity | coefficient | Γ in G eq 3.14; fixtures/pimpleFoam_v2006/Stokes.C:123-134 | high | ν_eff = ν = 1e-3 (Stokes) |
| LVS.C:102 | `dev2(...)` | **operator_argument** / argument | deviatoric_part_2 | algebraic | not in G (only dev, eq A.32 p.52); fixtures/pimpleFoam_v2006/upstream/TensorI.H:689-694 | high | A − ⅔tr(A)I (source: `t - 2*sph(t)`) |
| LVS.C:102 | `T(...)` | **operator_argument** / argument | tensor_transpose | algebraic | G §2.2 p.24 eq 2.2; Tab 2.2 p.26 | high | Pointwise transpose |
| LVS.C:103 | `fvm::laplacian(nuEff,U)` | operator / term | diffusion | implicit | G §3.4.1 eq 3.14; §1.1.3 p.12 | high | Textbook viscous diffusion (sign −1) |
| LVS.C:118 | `fvc::div(rho…dev2…)` | operator / term (not on path) | viscous_stress_divergence_explicit | explicit | G §3.4.5 eq 3.24 | medium | (rho,U) overload. Not called by pimpleFoam |
| LVS.C:118 | `fvc::grad(U)` | operator / argument (nested) | velocity_gradient | explicit | G §3.4.6 eq 3.25 | high | (rho,U) overload, not on the momentum path |
| LVS.C:119 | `fvm::laplacian(rho…,U)` | operator / term (not on path) | diffusion | implicit | G §3.4.1 eq 3.14 | high | Same overload as LVS.C:118, not on the momentum path |

### Borderline scope decisions

- **`UEqn.A()`, `UEqn.H()` and `UEqn.H1()` are `bookkeeping`.**
  - They add no new operator. They read the assembled matrix of terms that are already labeled. Source: A = D/V (fixtures/pimpleFoam_v2006/upstream/fvMatrix.C:771-795); H = (−Σ a_N U_N + source + boundary source)/V (fixtures/pimpleFoam_v2006/upstream/fvMatrix.C:800-858, line 834); H1 is the sum of the off-diagonal coefficients (fixtures/pimpleFoam_v2006/upstream/fvMatrix.C:862).
  - They still carry the key coupling: `H()` brings the explicit viscous term into `HbyA`, and so into the pressure equation. The graph must keep the `UEqn → A/H/H1 → rAU/HbyA/rAtU → pEqn` data-flow edges.
  - "Bookkeeping" means they are not a physics term. It does not mean they can be dropped.
- **`pEqn.flux()` is `bookkeeping`.** Numerically it is the face pressure-gradient flux of the Laplacian that was just solved. It is an accessor (fixtures/pimpleFoam_v2006/upstream/fvMatrix.C:910) of the operator at pEqn.H:46, not a new term.
- **`fvc::makeRelative`, `fvc::makeAbsolute` and `fvc::correctUf` are `bookkeeping`, even though they are `fvc::` calls.**
  - They change the frame of an existing flux using the mesh-motion flux φg (guide Table 3.1 p.35).
  - They are not Table 3.2 operators.
  - They are inactive here because the mesh is static.
- **`nuEff`, `dev2` and `T` are `operator_argument` (owner Rule 2).** They are a coefficient and pointwise tensor algebra inside the `fvc::div`/`fvm::laplacian` arguments, not discretized operators. Their roles are `coefficient` and `algebraic`. They are kept for Gate 1 identity matching and are never ROM terms.
- **`MRF.DDt(U)` and `fvOptions(U)` are `operator`**, as the brief specified. They add terms to the matrix.
- **All other `MRF.*` and `fvOptions.*` calls are `bookkeeping`:** constrain, correct, zeroFilter, makeRelative and correctBoundaryVelocity.

## Where the guide disagrees with existing decisions

No icoFoam **label** is contradicted by the guide. Three things need flagging:

1. **icoFoam GT header and notes (fixed 2026-09-17, citation only).** They said the `fvi::grad` and `fvi::div` entries were "confirmed against … Programmer's Guide v2512". The v2512 guide has no `fvi::` namespace at all. They now say the `fvi::` entries are confirmed against OpenFOAM-dev source only (`src/finiteVolume/finiteVolume/fvi/`: fviGrad.H, fviDiv.H), and that the guide does not cover `fvi::`. No label, argument, line or reviewed flag changed. The labels themselves (`pressure_gradient`, `flux_divergence`) agree with guide §1.1.3 and §3.4.5, applied through their `fvc::` equivalents.
2. **icoFoam `fvi::div` → `flux_divergence`.** The guide supports this label, but for a different reason than the one recorded. §3.4.5 separates "divergence" from "convection" by what the operator acts on: a face or vol field, "not the divergence of the product of a velocity and dependent variable". The recorded reason is that it appears in the pressure equation. The argument-based reason is the one that generalizes.
3. **`extractor/ontology.py` contradicts the guide.** This concerns the ontology, not the icoFoam GT.
   - It labels every `fvc::div` as `convection_explicit`, which contradicts §3.4.5 and Table 3.2 (Divergence, Exp, `div(chi)`, separate from Convection `div(psi, phi)`).
   - It labels every `fvc::grad` as `pressure_gradient_explicit`. The guide's gradient is field-agnostic (§3.4.6), and eq 3.3 shows the gradient of a vector.

Two smaller issues:
- **`flux_reconstruction` (icoFoam, kept here) is a naming tension, not a contradiction.** In OpenFOAM, `fvc::reconstruct` goes face→cell. `fvc::flux` goes cell→face (S_f·U_f, the F of guide eq 3.16). The guide's "Reconstruct" heading in §3.4.10 is empty.
- **`diffusion` for `fvm::laplacian(rAU|rAtU, p)` is consistent with the guide.** The guide names the operator "Laplacian" (eq 3.14) and uses it for "a transient diffusion equation" (§3.5.1). But the guide gives no physical reading for the pressure Laplacian.

## Ontology changes implied (proposals only, not implemented)

1. **Make `fvc::div` argument-aware.**
   - A single surfaceField argument, or any `phi*`/flux-typed argument, maps to `flux_divergence`.
   - `fvc::div(flux, vf)` with two arguments maps to `convection` (role explicit).
   - A volTensor/volSymmTensor argument maps to `stress_divergence_explicit`. Use `viscous_stress_divergence_explicit` when the argument contains `nuEff` or `dev`/`dev2`.
   - Delete `convection_explicit` as the default. Detecting the field type needs a declaration lookup (for example, `surfaceScalarField phiHbyA(...)`) or a name heuristic flagged as low confidence.
2. **Make `fvc::grad` argument-aware.** `p` (or `p_rgh`) maps to `pressure_gradient`. A velocity field (`U`, `this->U_`) maps to `velocity_gradient`. Anything else maps to a generic `gradient`.
3. **Rename `pressure_gradient_explicit` to `pressure_gradient`.** Explicitness is already in `discretization_role`. This also aligns with the reviewed icoFoam `fvi::grad` label.
4. **`fvc::snGrad`:** use `pressure_gradient_face_normal` for `p`, otherwise `surface_normal_gradient`. Drop its PROPOSED flag: guide §3.4.6 eq 3.28 and Table 3.2 cover it.
5. **`fvc::ddtCorr`:** keep `temporal_flux_correction`. Note the 2-argument (dev, icoFoam) and 3-argument v2006 `(U, phi, Uf)` overloads. Mark it `numerical: true`.
6. **Add `scope` (operator | operator_argument | bookkeeping) and `term_kind` (term | forwarding | argument) to `OperatorMeaning` / graph nodes.**
   - `term_kind` for a nested `fvc::` call is decided from the graph's `nested_in`. Nesting inside a bookkeeping wrapper does not demote a term.
   - Only `term_kind: term` nodes are ROM terms.
   - `nuEff`, `dev2`, `T` (and `dev`, `twoSymm`) are `operator_argument`.
   - Move `makeRelative`, `makeAbsolute` and `correctUf` to bookkeeping.
   - Add bookkeeping entries for unqualified member calls: `A`, `H`, `H1`, `flux`, `relax`, `setReference`, `solve`, `correctBoundaryConditions`, `constrainHbyA`, `constrainPressure`, `adjustPhi`, and `MRF.*`/`fvOptions.*` other than `DDt` and `operator()`.
7. **Add a receiver-keyed table for unqualified operator calls.**
   - `MRF.DDt` → `mrf_coriolis_source` (explicit).
   - `fvOptions(U)` → `user_source` (implicit_or_explicit).
   - `turbulence->divDevReff` → `viscous_stress_divergence` (implicit+explicit, expanded by dispatch.py).
   - `divDevRhoReff` → the same type. `divDevReff` and `divDevRhoReff` are both `term_kind: forwarding`.
8. **Add new `discretization_role` values:** `implicit+explicit` (composite dispatch), `implicit_or_explicit` (fvOptions, `fvm::SuSp`), `algebraic` (`dev2`, `T`, `dev`, `twoSymm`) and `coefficient` (`nuEff`).
9. **Add a `numerical` or `continuum` flag per node** (see the table below), so the ROM can separate physics terms from algorithmic terms.
10. **Allow a `diffusion` node to carry an `equation_role`** (momentum vs pressure_poisson), taken from graph context. This keeps the single-label icoFoam decision.

## Textbook NS vs parsed terms

The reference system is incompressible NS: ∂U/∂t + ∇·(UU) = −∇p + ∇·[ν(∇U+(∇U)ᵀ)] with ∇·U = 0. The table covers operator-scope entries only. "Active" refers to this case: laminar, no MRF, no fvOptions, static mesh, consistent yes, ddtCorr true.

| file:line | term | textbook NS term | continuum or numerical | active |
|---|---|---|---|---|
| UEqn.H:7 | fvm::ddt(U) | time derivative | continuum | yes |
| UEqn.H:7 | fvm::div(phi,U) | convection | continuum | yes |
| UEqn.H:8 | MRF.DDt(U) | none (rotating-frame Coriolis) | continuum, frame model | no |
| UEqn.H:9 | divDevReff(U) | viscous diffusion (forwarding; terms are LVS.C:102-103) | continuum | yes |
| UEqn.H:11 | fvOptions(U) | none (user source) | model | no |
| UEqn.H:21 | fvc::grad(p) | pressure gradient (predictor, lagged p) | continuum | yes |
| pEqn.H:3 | fvc::flux(HbyA) | none | numerical (face-flux construction) | yes |
| pEqn.H:7 | fvc::interpolate(rAU) | none | numerical | yes |
| pEqn.H:7 | fvc::ddtCorr | none | numerical (transient Rhie-Chow) | yes |
| pEqn.H:11 | fvc::interpolate(rAU) | none | numerical | no (dead branch) |
| pEqn.H:29 | fvc::interpolate(rAtU−rAU) | none | numerical (SIMPLEC) | yes |
| pEqn.H:29 | fvc::snGrad(p) | pressure gradient (face-normal part) | numerical (SIMPLEC consistency) | yes |
| pEqn.H:30 | fvc::grad(p) | pressure gradient | numerical (SIMPLEC consistency) | yes |
| pEqn.H:46 | fvm::laplacian(rAtU,p) | none directly; enforces continuity (pressure Poisson) | algorithmic (projection) | yes |
| pEqn.H:46 | fvc::div(phiHbyA) | continuity ∇·U (of the predicted flux) | algorithmic (projection) | yes |
| pEqn.H:64 | fvc::grad(p) | pressure gradient (corrector) | continuum | yes |
| ITM.C:117 | divDevRhoReff(U) | viscous diffusion (forwarding) | continuum | yes |
| LVS.C:87 | fvc::grad(U_) | none (stress output) | n/a | no (not on path) |
| LVS.C:102 | fvc::div(nuEff dev2(T(∇U))) | viscous diffusion (transpose/trace part) | continuum; zero in the continuum for constant ν and ∇·U=0, discretely nonzero | yes (small, not measured) |
| LVS.C:102 | fvc::grad(U) | viscous diffusion (inner gradient) | continuum | yes |
| LVS.C:102 | nuEff (operator_argument) | viscous diffusion coefficient ν | continuum | yes |
| LVS.C:102 | dev2, T (operator_argument) | viscous diffusion (tensor algebra) | continuum | yes |
| LVS.C:103 | fvm::laplacian(nuEff,U) | viscous diffusion (ν∇²U) | continuum | yes |
| LVS.C:118/119 | (rho,U) overload | viscous diffusion | continuum | no (not called) |

## Owner decisions (2026-09-17) and remaining questions

1. **Decided (Rule 1): operand-aware.** `snGrad(p)` → `pressure_gradient_face_normal`.
   - **Open:** the `fvm::laplacian(rAtU(), p)` → `diffusion` exception. Rule 1 would suggest an operand-aware pressure-Laplacian label, but the reviewed icoFoam key fixes `diffusion`. Keep the exception, or relabel both keys?
2. **Decided (Rule 2).** `nuEff`, `dev2` and `T` → `operator_argument`.
   - **Open:** confirm that the nested `fvc::grad(U)` calls (LVS.C:87/102/118) stay `scope: operator` with `term_kind: argument`.
3. **Decided (Rule 3).** Expanded. UEqn.H:9 and ITM.C:117 are `forwarding`. LVS.C:102 `fvc::div` and LVS.C:103 `fvm::laplacian` are `term`.
4. **Decided (Rule 4).** `user_source`, `implicit_or_explicit`.
   - **Open:** should `function` be `fvOptions` or `operator()`?
5. **Decided, done.** The icoFoam GT citation is fixed; labels are unchanged.
6. **Decided, done.** The upstream v2006 sources are in `fixtures/pimpleFoam_v2006/upstream/`, with blob ids in PROVENANCE.txt.

## Open items closed (2026-09-17, from the checked-in upstream sources)

- **`constrainPressure` is a no-op in this case.** `fixtures/pimpleFoam_v2006/upstream/constrainPressure.C:36-79` only calls `updateSnGrad` on `fixedFluxPressure` p patches. The overload used at pEqn.H:39 (96-107) forwards there. The case's `0.org/p` patches are zeroGradient (inlet, cylinder, top, bottom), fixedValue (outlet) and empty (front, back). The YAML `active_in_case` is updated.
- **The `experimentalDdtCorr` switch does not affect this case.** The switch defaults to 0 (`fixtures/pimpleFoam_v2006/upstream/ddtSchemeBase.C:34-37`, an OptimisationSwitch; `ddtScheme.H:74-76` says "Default is off"). The case controlDict does not set it. It also does not matter here: the incompressible Euler path `EulerDdtScheme.C:546` calls the 3-argument `fvcDdtPhiCoeff(U, phi, phiCorr)` (`ddtScheme.C:143-213`) directly, and that overload never reads the switch. Only the other overloads do (`ddtScheme.C:306, 330, 361`). So c_f = 1 − min(|φᵒ−S_f·U_fᵒ|/|φᵒ|, 1), and 0 on fixed-value U patches.
- **`fvc::makeRelative`, `fvc::makeAbsolute` and `fvc::correctUf` are confirmed no-ops on a static mesh.** makeRelative (`fixtures/pimpleFoam_v2006/upstream/fvcMeshPhi.C:76-86`) and makeAbsolute (`:115-125`) act only if `mesh.moving()`. correctUf (`:224-239`) acts only if `mesh.dynamic()`.
- **Still open:** `p.relax()` / `solution::relaxField`, which was not fetched. Also, whether the dead else-branch at pEqn.H:11 is an upstream bug.
