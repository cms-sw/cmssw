#!/bin/bash
# Regenerate a BL-fit material map from scratch, for any geometry, under its catalog name
# <outdir>/BLMaterialMap_<tag>_BP<beam-pipe>_v<version>.bin (version: $MAPVERSION, default 1):
#   rays (blMaterialMapRays_cfg.py, one cmsRun per job, + the run's G4 material table) -> lattice
#   (blMaterialMapBuild) -> map (blMaterialMapEmit.py: the 1/X0 lattice and the dE/dx lattice, byte format
#   of BLMaterialMapFile.h, with the geometry fingerprint in the header and this run's provenance embedded)
#
#   blMaterialMapRun.sh <outdir> [njobs] [rays-per-job] [tag] [geometry-cff] [era]
#   blMaterialMapRun.sh --fingerprint-only <outdir> [tag] [geometry-cff] [era]
#
# Run in a CMSSW environment in which RecoTracker/PixelTrackFitting has been built. The defaults are the
# shipped recipe: T35 (D121). Every job also dumps the geometry fingerprint (BLMaterialMapFingerprintDump)
# into the trees directory; the map is built only if all jobs agree on it (the geometry was uniform), and
# written into the map's header, so readers reject a map made for another geometry. The second form runs
# a single 1-event job and writes only PROVENANCE.txt: the geometry key of a geometry, before paying for
# the rays.
set -eu
USAGE="usage: blMaterialMapRun.sh <outdir> [njobs] [rays-per-job] [tag] [geometry-cff] [era]
       blMaterialMapRun.sh --fingerprint-only <outdir> [tag] [geometry-cff] [era]"
FPONLY=0
if [ "${1:-}" = --fingerprint-only ]; then FPONLY=1; shift; fi
OUT=${1:?$USAGE}
if [ "$FPONLY" = 1 ]; then
  NJ=1
  NEV=1
  TAG=${2:-T35}
  GEOM=${3:-Configuration.Geometry.GeometryExtendedRun4D121Reco_cff}
  ERA=${4:-Phase2C22I13M9}
  RAYS="1 event, fingerprint-only"
else
  NJ=${2:-40}
  NEV=${3:-150000}
  TAG=${4:-T35}
  GEOM=${5:-Configuration.Geometry.GeometryExtendedRun4D121Reco_cff}
  ERA=${6:-Phase2C22I13M9}
  RAYS="$NJ jobs x $NEV generated rays, seed 1..$NJ, eta flat in [-6,6], phi flat in [-pi,pi]"
fi
HERE=$(cd "$(dirname "$0")" && pwd)
if [ "$FPONLY" = 0 ]; then
  BUILD=$(command -v blMaterialMapBuild || echo "${CMSSW_BASE:-}/test/${SCRAM_ARCH:-}/blMaterialMapBuild")
  [ -x "$BUILD" ] || { echo "blMaterialMapBuild not found (cmsenv done?): build RecoTracker/PixelTrackFitting first" >&2; exit 1; }
fi

mkdir -p "$OUT/trees" "$OUT/bins" "$OUT/logs"
# The provenance the map header is written from: the beam-pipe and material descriptions are read off the
# geometry configuration itself, and the beam-pipe tag (e.g. 2030/v3) off the beam-pipe XML's path.
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
BP=$(echo "$XML" | sed -n '1s/.*beampipe\/\([0-9][0-9]*\/v[0-9][0-9]*\)\/.*/\1/p')
[ -n "$BP" ] || { echo "no beampipe/<year>/v<n>/ XML in $GEOM: $(echo "$XML" | sed -n 1p)" >&2; exit 1; }
{
  echo "date      : $(date -Is)"
  echo "host      : $(hostname)"
  echo "release   : ${CMSSW_VERSION:-unknown} ${SCRAM_ARCH:-unknown}"
  echo "geometry  : $GEOM"
  echo "beam pipe : $(echo "$XML" | sed -n 1p)"
  echo "materials : $(echo "$XML" | sed -n 2p)"
  echo "era       : $ERA"
  echo "rays      : $RAYS"
} > "$OUT/PROVENANCE.txt"
cat "$OUT/PROVENANCE.txt"

# MaterialBudgetAction opens its tree file relative to the working directory, so every job runs inside
# the trees directory and gets bare file names; likewise the material table and the fingerprint dump.
if [ "$FPONLY" = 1 ]; then
  (cd "$OUT/trees" && cmsRun "$HERE/blMaterialMapRays_cfg.py" nEvents=1 seed=1 out=fp.root \
    materials=fp_mat.txt fingerprint=fingerprint_001.txt dumpPositions=1 geometry="$GEOM" era="$ERA" \
    > "$OUT/logs/fingerprint.log" 2>&1)
else
  for ((j = 1; j <= NJ; j++)); do
    n=$(printf %03d "$j")
    # job 001 also dumps the sensor positions: the emitter appends them to the map as its geometry
    # reference (every job sees the same geometry; one dump is enough)
    DP=0; [ "$j" -eq 1 ] && DP=1
    (cd "$OUT/trees" && cmsRun "$HERE/blMaterialMapRays_cfg.py" nEvents="$NEV" seed="$j" out="rays_$n.root" \
      materials="materials_$n.txt" fingerprint="fingerprint_$n.txt" dumpPositions=$DP geometry="$GEOM" era="$ERA" \
      > "$OUT/logs/rays_$n.log" 2>&1) &
  done
  wait
  NT=$(ls "$OUT"/trees/rays_*.root 2>/dev/null | wc -l)
  [ "$NT" -eq "$NJ" ] || { echo "only $NT of $NJ step trees were written; see $OUT/logs" >&2; exit 1; }
  echo "rays done: $NT trees"
fi

# Every job dumped the geometry fingerprint; all of them must be the same, else the jobs did not all see
# the same geometry and the map must not be built from these trees.
NF=$(ls "$OUT"/trees/fingerprint_*.txt 2>/dev/null | wc -l)
[ "$NF" -eq "$NJ" ] || { echo "only $NF of $NJ fingerprint dumps were written; see $OUT/logs" >&2; exit 1; }
[ "$(awk 'FNR == 1' "$OUT"/trees/fingerprint_*.txt | sort -u | wc -l)" -eq 1 ] ||
  { echo "the $NJ jobs disagree on the geometry fingerprint; see $OUT/trees/fingerprint_*.txt" >&2; exit 1; }
FPLINE=$(awk 'FNR == 1' "$OUT"/trees/fingerprint_*.txt | sort -u)
case $FPLINE in
  "FINGERPRINT 0x"*) FP=${FPLINE#FINGERPRINT } ;;
  *) echo "unexpected fingerprint line: $FPLINE (BLMaterialMapFingerprintDump did not run?)" >&2; exit 1 ;;
esac
FPFILE=$(ls "$OUT"/trees/fingerprint_*.txt | sed -n 1p)
SENSORS=$(awk '$1 == "SENSORS" {s = s " " $3} END {print substr(s, 2)}' "$FPFILE")
{
  echo "fingerprint: $FP"
  echo "sensors   : $SENSORS"
} >> "$OUT/PROVENANCE.txt"
cat "$OUT/PROVENANCE.txt"

if [ "$FPONLY" = 1 ]; then
  echo "fingerprint of $GEOM: $FP (sensors: $SENSORS)"
  exit 0
fi

# every job wrote the same material table; the builder reads the first
MAT=$OUT/trees/materials_001.txt
[ -s "$MAT" ] || { echo "no material table $MAT (BLMaterialTableDump did not run)" >&2; exit 1; }
ls "$OUT"/trees/rays_*.root | xargs -P 8 -I{} bash -c 'b=$(basename {} .root); "'"$BUILD"'" "'"$OUT"'/bins/$b.bin" "'"$MAT"'" {} > "'"$OUT"'/logs/build_$b.log" 2>&1'
echo "lattice done: $(ls "$OUT"/bins/*.bin | wc -l) accumulators"

VER=${MAPVERSION:-1}
BIN=$OUT/BLMaterialMap_${TAG}_BP${BP//\//}_v$VER.bin
POS=$OUT/trees/fingerprint_001.txt
[ -s "$POS" ] || { echo "no position dump $POS" >&2; exit 1; }
python3 "$HERE/blMaterialMapEmit.py" --bins "$OUT/bins" --provenance "$OUT/PROVENANCE.txt" --tag "$TAG" \
  --beam-pipe "$BP" --fingerprint "$FP" --positions "$POS" --map-version "$VER" --out "$BIN" \
  --index "$OUT/BLMaterialMap.index"
echo "map: $BIN; index line: $OUT/BLMaterialMap.index -- copy the map to \
RecoTracker/PixelSeeding/data/BLMaterialMap/ and merge the line into its BLMaterialMap.index (README)"
