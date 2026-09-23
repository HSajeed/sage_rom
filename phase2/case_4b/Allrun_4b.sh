#!/bin/sh
# Step 4b treatment run: two copies of the dataset case, baseline
# (pimpleFoam, unmodified) and drag (pimpleDragFoam, with the injected
# fvm::Sp(cD*mag(U), U) UEqn term), same mesh, same controlDict/fvSchemes/
# fvSolution, same 0.org initial/boundary conditions. See README.md for
# the cD derivation and what "same" means precisely.
#
# UNTESTED: written and reviewed, never run -- no local OpenFOAM v2006
# install (see README.md "Status"). Run this yourself with
#   ./Allrun_4b.sh [<dataset_case_dir>] [<work_dir>]
# defaults: <dataset_case_dir>=$FLOWTORCH_DATASETS/of_cylinder2D_binary,
# <work_dir>=./work_4b (created here, next to this script).
#
# This does NOT re-mesh: the dataset case already ships a built mesh
# (constant/polyMesh/, produced by blockMesh + snappyHexMesh -overwrite +
# extrudeMesh per the dataset's own Allrun) -- both arms reuse it verbatim
# by copying constant/. If you're instead starting from an unmeshed case,
# run the dataset's own Allrun's mesh block first (blockMesh;
# snappyHexMesh -overwrite; extrudeMesh) and point this script at that.

set -e
cd "${0%/*}" || exit
. "${WM_PROJECT_DIR:?}/bin/tools/RunFunctions"

DATASET_CASE="${1:-${FLOWTORCH_DATASETS:?}/of_cylinder2D_binary}"
WORK_DIR="${2:-$PWD/work_4b}"

# cD derivation: see README.md "cD derivation" for the full arithmetic.
# D=0.1 m, U_mean=1.0 m/s, nu=1e-3 (Re=100, Schafer-Turek 2D-1 benchmark
# geometry the dataset uses). Convective term ~ U^2/D = 10 m/s^2. Target
# drag ~ 2-5% of that => cD*U_mean^2 in [0.2, 0.5] m/s^2 => cD in
# [0.2, 0.5] 1/m (since U_mean=1.0). Chosen: cD = 0.3 1/m (~3%).
CD_VALUE="${CD_VALUE:-0.3}"

if [ ! -d "$DATASET_CASE/constant/polyMesh" ]; then
    echo "Allrun_4b.sh: $DATASET_CASE/constant/polyMesh not found -- point \$1 " >&2
    echo "at a case with a built mesh (see this script's header comment)." >&2
    exit 1
fi

echo "Allrun_4b.sh: building pimpleDragFoam"
( cd ../pimpleDragFoam && wmake )

mkdir -p "$WORK_DIR"
BASELINE_DIR="$WORK_DIR/baseline"
DRAG_DIR="$WORK_DIR/drag"

make_case_copy () {
    # $1 = destination case dir. Copies mesh (constant/), case
    # configuration (system/) and the ROI/BC template (0.org/) only --
    # deliberately NOT any numbered time directory or log.* from the
    # dataset case (those are simulation OUTPUT, not case setup).
    dest="$1"
    rm -rf "$dest"
    mkdir -p "$dest"
    cp -r "$DATASET_CASE/constant" "$dest/"
    cp -r "$DATASET_CASE/system" "$dest/"
    cp -r "$DATASET_CASE/0.org" "$dest/"
    touch "$dest/post.foam"
}

echo "Allrun_4b.sh: assembling baseline case at $BASELINE_DIR"
make_case_copy "$BASELINE_DIR"

echo "Allrun_4b.sh: assembling drag case at $DRAG_DIR"
make_case_copy "$DRAG_DIR"
# Append cD (dimensions [0 -1 0 0 0 0 0], see
# ../pimpleDragFoam/createFields.H) to the drag case's transportProperties
# -- same nu as the dataset, one extra line.
cat >> "$DRAG_DIR/constant/transportProperties" <<EOF

cD              [0 -1 0 0 0 0 0] $CD_VALUE;
EOF
# Point this case's controlDict at pimpleDragFoam instead of pimpleFoam.
sed -i.bak 's/^application[[:space:]]\+pimpleFoam;/application     pimpleDragFoam;/' \
    "$DRAG_DIR/system/controlDict"
rm -f "$DRAG_DIR/system/controlDict.bak"

for pair in "$BASELINE_DIR:pimpleFoam" "$DRAG_DIR:pimpleDragFoam"; do
    CASE_DIR="${pair%%:*}"
    APP="${pair##*:}"
    (
        cd "$CASE_DIR"
        # Same inlet-velocity setup as the dataset's own Allrun.
        cp -r 0.org 0
        runApplication setExprBoundaryFields
        runApplication "$APP"
    )
    echo "Allrun_4b.sh: $APP finished in $CASE_DIR (see $CASE_DIR/log.$APP)"
done

echo "Allrun_4b.sh: done. Score with (from phase2/):"
echo "  python -m static_rom.run_step4a --data-dir $BASELINE_DIR --source-dir fixtures/pimpleFoam_v2006 --case-dir $BASELINE_DIR --out results_step4b_baseline"
echo "  python -m static_rom.run_step4a --data-dir $DRAG_DIR --source-dir fixtures/pimpleFoam_v2006_drag --case-dir $DRAG_DIR --out results_step4b_drag"
echo "  python -m static_rom.run_step4a --data-dir $BASELINE_DIR --source-dir fixtures/pimpleFoam_v2006 --case-dir $BASELINE_DIR --placebo-drag --out results_step4b_placebo"

#------------------------------------------------------------------------------
