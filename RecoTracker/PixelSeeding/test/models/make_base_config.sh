#!/bin/bash
# =============================================================================
# make_base_config.sh -- generate the base reconstruction config the training
# dumps run on top of.
# =============================================================================
# Both dump configs in this directory (trackNano_config.py, triplet_dump_cfg.py)
# only ADD tables to an existing reconstruction process; they do not build one.
# That base process ("chassis") is an ordinary cmsDriver.py output. This script
# writes one, so a fresh checkout can produce training datasets without any
# private config file.
#
#   ./make_base_config.sh --files <input.root>[,<more.root>...] --out <path.py>
#
# The generated module is imported by the dump configs through NANO_CHASSIS /
# DS_CHASSIS (its directory must be on PYTHONPATH); the retrain entry scripts do
# that for you and reuse the file across steps.
#
# OPTIONS
#   --files LIST   comma-separated input files, a directory, a glob, or a text
#                  file with one path per line                      (required)
#   --out PATH     output config file                    (default: ./base_reco_cfg.py)
#   --menu NAME    HLT menu                              (default: 75e33_trackingOnly)
#                  The tracking-only menu runs the whole pixel+OT chain the
#                  training data comes from (both CA iterations, the stubs
#                  merger, both high-purity selectors) and skips everything
#                  else, so it is much cheaper per event than the full menu.
#   --era NAME     (default: Phase2C22I13M9)
#   --geometry G   (default: ExtendedRun4D121)
#   --conditions C (default: auto:phase2_realistic_T35)
#   --modifiers M  --procModifiers, verbatim                (default: ngtScouting)
#                  ngtScouting = phase2CAStubs + pixelTrackMask + ngtScoutingBase:
#                  the two-iteration stub CA runs unchanged and hltGeneralTracks
#                  becomes a pass-through RecoTrackSelector over the pixel tracks,
#                  so no iterative tracking runs and a dump is cheap. The pixel
#                  chain every model here is trained on is unaffected.
#   --events N     maxEvents baked into the config; the dump configs override it
#                  from their own event count                       (default: 10)
#   --input-commands S
#                  cmsDriver --inputCommands string. The default drops the
#                  products of a previous HLT process that RelVal inputs carry,
#                  so unqualified InputTags cannot resolve ambiguously against
#                  them. Pass an empty string to omit the option entirely.
#                  (default: "keep *, drop *_hlt*_*_HLT,
#                             drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT")
#   --force        overwrite an existing output file
#
# NOTE ON --output={}. The generated chassis is a base for the dump configs,
# which add their own NanoAOD output. The cmsDriver call therefore passes
# '--output={}' so ConfigBuilder emits NO PoolOutputModule and no
# <eventcontent>output_step EndPath; without it every dump job would silently
# write a second, full GEN-SIM-DIGI-RAW file next to its table.
# =============================================================================
set -euo pipefail

FILES=""
OUT="base_reco_cfg.py"
MENU="75e33_trackingOnly"
ERA="Phase2C22I13M9"
GEOM="ExtendedRun4D121"
COND="auto:phase2_realistic_T35"
MODIFIERS="ngtScouting"
NEVT=10
FORCE=0
# Stop unqualified InputTags resolving against products of an earlier HLT process on the input.
INPUT_CMDS="keep *, drop *_hlt*_*_HLT, drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT"

while [ $# -gt 0 ]; do
  case "$1" in
    --files)      FILES="${2:?--files needs a value}"; shift 2 ;;
    --out)        OUT="${2:?--out needs a value}"; shift 2 ;;
    --menu)       MENU="${2:?}"; shift 2 ;;
    --era)        ERA="${2:?}"; shift 2 ;;
    --geometry)   GEOM="${2:?}"; shift 2 ;;
    --conditions) COND="${2:?}"; shift 2 ;;
    --modifiers)  MODIFIERS="${2:?}"; shift 2 ;;
    --events)     NEVT="${2:?}"; shift 2 ;;
    --input-commands|--inputCommands) INPUT_CMDS="${2-}"; shift 2 ;;
    --force)      FORCE=1; shift ;;
    -h|--help)    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d'; exit 0 ;;
    *) echo "make_base_config.sh: unknown option '$1'"; exit 1 ;;
  esac
done

[ -n "$FILES" ] || { echo "make_base_config.sh: --files is required"; exit 1; }

if [ -e "$OUT" ] && [ "$FORCE" -ne 1 ]; then
  echo "make_base_config.sh: $OUT already exists (use --force to regenerate)"
  exit 0
fi

# Normalise --files into the comma-separated 'file:<path>' list cmsDriver wants.
# Accepted: comma list, a directory of .root files, a shell glob, or a text file
# with one path per line.
expand_files() {
  local spec="$1" list=""
  if [ -d "$spec" ]; then
    list=$(ls -1 "$spec"/*.root 2>/dev/null)
  elif [ -f "$spec" ] && [[ "$spec" != *.root ]]; then
    list=$(grep -vE '^[[:space:]]*(#|$)' "$spec" | grep -E '\.root$')
  else
    list=$(echo "$spec" | tr ',' '\n' | grep -E '\.root$')
  fi
  # Local paths need the file: prefix; catalogue paths (/store/...) and paths
  # that already carry a prefix are passed through unchanged.
  echo "$list" | awk 'NF { if ($0 ~ /^file:/ || $0 ~ /^\/store\//) print $0; else print "file:" $0 }' | paste -sd,
}

FILEIN=$(expand_files "$FILES")
[ -n "$FILEIN" ] || { echo "make_base_config.sh: no input files matched '$FILES'"; exit 1; }

command -v cmsDriver.py >/dev/null 2>&1 || {
  echo "make_base_config.sh: cmsDriver.py not found -- source the CMSSW environment first"; exit 1; }

mkdir -p "$(dirname "$OUT")"
echo ">>> generating the base reconstruction config: $OUT"
echo "    menu=$MENU era=$ERA geometry=$GEOM conditions=$COND modifiers=$MODIFIERS"
echo "    no output module (--output={}); inputCommands=${INPUT_CMDS:-<none>}"
DRIVER_ARGS=(step2
  -s "L1P2GT,HLT:$MENU"
  --processName HLTX
  --conditions "$COND"
  --geometry "$GEOM"
  --era "$ERA"
  --procModifiers "$MODIFIERS"
  --eventcontent FEVTDEBUGHLT
  --datatier GEN-SIM-DIGI-RAW
  --filein "$FILEIN"
  -n "$NEVT"
  --no_exec
  # No output module: the dump configs add their own. See the note in the header.
  '--output={}'
  --python_filename "$OUT")
if [ -n "$INPUT_CMDS" ]; then DRIVER_ARGS+=("--inputCommands=$INPUT_CMDS"); fi
cmsDriver.py "${DRIVER_ARGS[@]}" >/dev/null

echo ">>> wrote $OUT"
echo "    use it with:  NANO_CHASSIS=$(basename "${OUT%.py}")  (or DS_CHASSIS for the triplet dump)"
echo "    and PYTHONPATH=$(cd "$(dirname "$OUT")" && pwd)"
