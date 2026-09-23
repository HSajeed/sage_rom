Info<< "Reading field p\n" << endl;
volScalarField p
(
    IOobject
    (
        "p",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

#include "createPhi.H"


label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p, pimple.dict(), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());


singlePhaseTransportModel laminarTransport(U, phi);

// Step 4b addition (NOT upstream v2006 -- see PROVENANCE.txt): quadratic
// drag coefficient for the injected fvm::Sp(cD*mag(U), U) UEqn term.
// Dimensions [0 -1 0 0 0 0 0] (1/length): cD*mag(U) is then 1/time, so
// fvm::Sp(cD*mag(U), U) matches the other UEqn terms' m/s^2 (kinematic,
// rho=1) dimensions. Read from constant/transportProperties alongside nu;
// see case_4b/README.md "cD derivation" for the numeric value.
dimensionedScalar cD
(
    "cD",
    dimless/dimLength,
    laminarTransport
);

autoPtr<incompressible::turbulenceModel> turbulence
(
    incompressible::turbulenceModel::New(U, phi, laminarTransport)
);

#include "createMRF.H"
#include "createFvOptions.H"
