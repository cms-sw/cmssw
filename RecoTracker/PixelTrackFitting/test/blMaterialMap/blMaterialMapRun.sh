#!/bin/bash
# Regenerate the BL-fit material table src/BLMaterialMap<tag>.cc from scratch, for any geometry:
#   rays (blMaterialMapRays_cfg.py, one cmsRun per job) -> lattice (blMaterialMapBuild) -> table (blMaterialMapEmit.py)
#
#   blMaterialMapRun.sh <outdir> [njobs] [rays-per-job] [tag] [geometry-cff] [era]
#
# Run in a CMSSW environment in which RecoTracker/PixelTrackFitting has been built. The defaults are the recipe
# of the shipped table; a run in the release that made it reproduces it value for value (the emitter reports
# the comparison when src/BLMaterialMap<tag>.cc exists). To ship the result: copy <outdir>/BLMaterialMap<tag>.cc
# to src/ (one table file must be compiled in: remove the previous one) and rebuild. Nothing else changes.
set -eu
OUT=${1:?usage: blMaterialMapRun.sh <outdir> [njobs] [rays-per-job] [tag] [geometry-cff] [era]}
NJ=${2:-40}
NEV=${3:-150000}
TAG=${4:-D121}
GEOM=${5:-Configuration.Geometry.GeometryExtendedRun4D121Reco_cff}
ERA=${6:-Phase2C22I13M9}
HERE=$(cd "$(dirname "$0")" && pwd)
SHIPPED=$HERE/../../src/BLMaterialMap$TAG.cc
BUILD=$(command -v blMaterialMapBuild || echo "$CMSSW_BASE/test/$SCRAM_ARCH/blMaterialMapBuild")
[ -x "$BUILD" ] || { echo "blMaterialMapBuild not found: build RecoTracker/PixelTrackFitting first" >&2; exit 1; }

mkdir -p "$OUT/trees" "$OUT/bins" "$OUT/logs"
# The provenance the table header is written from: the beam-pipe and material descriptions are read off the
# geometry configuration itself.
XML=$(python3 - "$GEOM" <<'PY'
import sys
import FWCore.ParameterSet.Config as cms
p = cms.Process("X")
p.load(sys.argv[1])
files = list(p.XMLIdealGeometryESSource.geomXMLFiles)
print("|".join(f for f in files if "beampipe" in f.lower()))
print("|".join(f for f in files if "/materials/" in f.lower()))
PY
)
{
  echo "date      : $(date -Is)"
  echo "host      : $(hostname)"
  echo "release   : ${CMSSW_VERSION:-unknown} ${SCRAM_ARCH:-unknown}"
  echo "geometry  : $GEOM"
  echo "beam pipe : $(echo "$XML" | sed -n 1p)"
  echo "materials : $(echo "$XML" | sed -n 2p)"
  echo "era       : $ERA"
  echo "rays      : $NJ jobs x $NEV generated rays, seed 1..$NJ, eta flat in [-6,6], phi flat in [-pi,pi]"
} > "$OUT/PROVENANCE.txt"
cat "$OUT/PROVENANCE.txt"

# MaterialBudgetAction opens its tree file relative to the working directory, so every job runs inside
# the trees directory and gets a bare file name.
for ((j = 1; j <= NJ; j++)); do
  n=$(printf %03d "$j")
  (cd "$OUT/trees" && cmsRun "$HERE/blMaterialMapRays_cfg.py" nEvents="$NEV" seed="$j" out="rays_$n.root" \
    geometry="$GEOM" era="$ERA" > "$OUT/logs/rays_$n.log" 2>&1) &
done
wait
NT=$(ls "$OUT"/trees/rays_*.root 2>/dev/null | wc -l)
[ "$NT" -eq "$NJ" ] || { echo "only $NT of $NJ step trees were written; see $OUT/logs" >&2; exit 1; }
echo "rays done: $NT trees"

ls "$OUT"/trees/rays_*.root | xargs -P 8 -I{} bash -c 'b=$(basename {} .root); "'"$BUILD"'" "'"$OUT"'/bins/$b.bin" {} > "'"$OUT"'/logs/build_$b.log" 2>&1'
echo "lattice done: $(ls "$OUT"/bins/*.bin | wc -l) accumulators"

if [ -f "$SHIPPED" ]; then
  python3 "$HERE/blMaterialMapEmit.py" --bins "$OUT/bins" --provenance "$OUT/PROVENANCE.txt" --tag "$TAG" \
    --out "$OUT/BLMaterialMap$TAG.cc" --check "$SHIPPED" || true
else
  python3 "$HERE/blMaterialMapEmit.py" --bins "$OUT/bins" --provenance "$OUT/PROVENANCE.txt" --tag "$TAG" \
    --out "$OUT/BLMaterialMap$TAG.cc"
fi
echo "table: $OUT/BLMaterialMap$TAG.cc (copy to RecoTracker/PixelTrackFitting/src/ and rebuild)"
