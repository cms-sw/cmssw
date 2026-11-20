#!/bin/bash
# Regenerate a material map of the BrokenLine fit as <outdir>/BLMaterialMap_<tag>_BP<beam-pipe>_v<version>.bin:
# rays (blMaterialMapRays_cfg.py) -> lattice sums (blMaterialMapBuild) -> map (blMaterialMapEmit.py).
#
#   blMaterialMapRun.sh <outdir> [njobs] [rays-per-job] [tag] [geometry-cff] [era] [map-version]
#   blMaterialMapRun.sh --fingerprint-only <outdir> [tag] [geometry-cff] [era]
#
# Run in a CMSSW area where RecoTracker/PixelTrackFitting is built. The defaults give the shipped T35 (D121) map.
# The second form runs one event and writes only the geometry fingerprint and PROVENANCE.txt (see README.md).
set -eu
USAGE="usage: blMaterialMapRun.sh <outdir> [njobs] [rays-per-job] [tag] [geometry-cff] [era] [map-version]
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
  VER=${7:-1}
  RAYS="$NJ jobs x $NEV generated rays, seed 1..$NJ, eta flat in [-6,6], phi flat in [-pi,pi]"
fi
HERE=$(cd "$(dirname "$0")" && pwd)
if [ "$FPONLY" = 0 ]; then
  BUILD=$(command -v blMaterialMapBuild || echo "${CMSSW_BASE:-}/test/${SCRAM_ARCH:-}/blMaterialMapBuild")
  [ -x "$BUILD" ] || { echo "blMaterialMapBuild not found (cmsenv done?): build RecoTracker/PixelTrackFitting first" >&2; exit 1; }
fi

mkdir -p "$OUT/trees" "$OUT/bins" "$OUT/logs"
# The provenance: the beam-pipe and material descriptions come from the geometry configuration, and the
# beam-pipe tag (e.g. 2030/v3) from the path of the beam-pipe XML.
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

# Every job runs in the trees directory, because MaterialBudgetAction opens its file there.
if [ "$FPONLY" = 1 ]; then
  (cd "$OUT/trees" && cmsRun "$HERE/blMaterialMapRays_cfg.py" nEvents=1 seed=1 out=fp.root \
    materials=fp_mat.txt fingerprint=fingerprint_001.txt dumpPositions=1 geometry="$GEOM" era="$ERA" \
    > "$OUT/logs/fingerprint.log" 2>&1)
else
  PIDS=()
  for ((j = 1; j <= NJ; j++)); do
    n=$(printf %03d "$j")
    # job 001 also writes the sensor positions for the map
    DP=0; [ "$j" -eq 1 ] && DP=1
    (cd "$OUT/trees" && cmsRun "$HERE/blMaterialMapRays_cfg.py" nEvents="$NEV" seed="$j" out="rays_$n.root" \
      materials="materials_$n.txt" fingerprint="fingerprint_$n.txt" dumpPositions=$DP geometry="$GEOM" era="$ERA" \
      > "$OUT/logs/rays_$n.log" 2>&1) &
    PIDS+=($!)
  done
  NFAILED=0
  for p in "${PIDS[@]}"; do
    wait "$p" || NFAILED=$((NFAILED + 1))
  done
  [ "$NFAILED" -eq 0 ] || { echo "$NFAILED of $NJ ray jobs failed; see $OUT/logs" >&2; exit 1; }
  NT=$(ls "$OUT"/trees/rays_*.root 2>/dev/null | wc -l)
  [ "$NT" -eq "$NJ" ] || { echo "only $NT of $NJ step trees were written; see $OUT/logs" >&2; exit 1; }
  echo "rays done: $NT trees"
fi

# All jobs must write the same geometry fingerprint.
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

# the builder reads the union of the material tables of the jobs
MAT=$OUT/trees/materials_all.txt
awk '{ material = $0; sub(/^[^ ]+ /, "", material) } !seen[material]++' "$OUT"/trees/materials_[0-9]*.txt > "$MAT"
[ -s "$MAT" ] || { echo "no material table $MAT (BLMaterialTableDump did not run)" >&2; exit 1; }
ls "$OUT"/trees/rays_*.root | xargs -P 8 -I{} bash -c 'b=$(basename {} .root); "'"$BUILD"'" "'"$OUT"'/bins/$b.bin" "'"$MAT"'" {} > "'"$OUT"'/logs/build_$b.log" 2>&1' ||
  { echo "a lattice job failed; see $OUT/logs" >&2; exit 1; }
echo "lattice done: $(ls "$OUT"/bins/*.bin | wc -l) accumulators"

BIN=$OUT/BLMaterialMap_${TAG}_BP${BP//\//}_v$VER.bin
POS=$OUT/trees/fingerprint_001.txt
[ -s "$POS" ] || { echo "no position dump $POS" >&2; exit 1; }
python3 "$HERE/blMaterialMapEmit.py" --bins "$OUT/bins" --provenance "$OUT/PROVENANCE.txt" --tag "$TAG" \
  --beam-pipe "$BP" --fingerprint "$FP" --positions "$POS" --map-version "$VER" --out "$BIN" \
  --index "$OUT/BLMaterialMap.index"
echo "map: $BIN; index line: $OUT/BLMaterialMap.index -- copy the map to \
RecoTracker/PixelSeeding/data/BLMaterialMap/ and merge the line into its BLMaterialMap.index (README)"
