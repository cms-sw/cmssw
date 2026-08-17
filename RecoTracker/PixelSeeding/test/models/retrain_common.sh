# =============================================================================
# retrain_common.sh -- shared helpers for the retraining scripts.
# =============================================================================
# Sourced by retrain_prompt.sh / retrain_displaced.sh / retrain_merged.sh and by
# the two per-stage workers (retrain_triplet_gate.sh, retrain_track_dnn.sh). Not
# executable on its own.
#
# It provides: one option parser for all the scripts, the CMSSW environment
# setup, base-config (chassis) resolution, the dataset dump commands, lookups of
# the values the deployed configuration currently carries, the shared part of
# the final-selector forest training, a memory watchdog, and the "what was
# produced / where it goes / what to update" report every artifact-producing
# step ends with.
# =============================================================================

# ---- locations --------------------------------------------------------------
# This file lives at src/RecoTracker/PixelSeeding/test/models/, so the CMSSW src
# directory is four levels up. Nothing here uses an absolute path from outside
# the checkout.
RT_MODELS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RT_SRC="$(cd "$RT_MODELS/../../../.." && pwd)"
RT_PLUGINS="$RT_SRC/RecoTracker/PixelSeeding/plugins/alpaka"
RT_DATA="$RT_SRC/RecoTracker/FinalTrackSelectors/data/PixelTrackTorchHighPuritySelector"
RT_CFI="$RT_SRC/HLTrigger/Configuration/python/HLT_75e33/modules"
RT_DUMP_MACRO="$RT_PLUGINS/CATripletDumpMacro.h"

# Per-chain configuration files whose values the scripts read back, so that a
# retrain always reports against what the chain is actually running.
rt_ca_cfi() {  # $1 = prompt|displaced
  if [ "$1" = prompt ]; then echo "$RT_CFI/hltPhase2PixelTracksSoAWithStubs_cfi.py"
  else echo "$RT_CFI/hltPhase2PixelTracksSoADisplacedWithStubs_cfi.py"; fi
}
rt_hp_cfi() {  # $1 = prompt|displaced|merged
  if [ "$1" = merged ]; then echo "$RT_CFI/hltPhase2PixelTrackHighPuritySelectorMerged_cfi.py"; return 0; fi
  if [ "$1" = prompt ]; then
    # The prompt selector lives under the upstream label in menus that keep the
    # Torch baseline and swap the forest in with a phase2CAStubs toReplaceWith,
    # and under its own label in menus that do not. Take whichever is there.
    local f
    for f in hltPhase2PixelTrackHighPuritySelector_cfi.py \
             hltPhase2PixelTrackTorchHighPuritySelector_cfi.py; do
      if [ -f "$RT_CFI/$f" ]; then echo "$RT_CFI/$f"; return 0; fi
    done
    echo "$RT_CFI/hltPhase2PixelTrackHighPuritySelector_cfi.py"
  else echo "$RT_CFI/hltPhase2PixelTrackHighPuritySelectorDisplaced_cfi.py"; fi
}

rt_die() { echo "ERROR: $*" >&2; exit 1; }

# ---- option parsing ---------------------------------------------------------
# One parser for every script in this directory. The first bare word is the
# step; everything else is a long option. Environment variables of the same
# meaning (WORKDIR, SAMPLES, NEVT, THREADS, DEVICE) are still honoured as
# defaults so existing habits keep working.
RT_STEP=""
RT_FOR=""
RT_WORK="${WORKDIR:-}"
RT_INPUT=""
# Per-sample inputs. RT_INPUT_ORDER holds every --input occurrence in the
# order it was given; RT_INPUT_NAMES/RT_INPUT_SPECS hold the 'name=spec' pairs.
# Plain bash arrays rather than an associative one so the file stays sourceable
# by any bash the checkout is built with.
RT_INPUT_ORDER=()
RT_INPUT_NAMES=()
RT_INPUT_SPECS=()
RT_SAMPLE=""
RT_EVENTS=""
RT_DATASETS=()
RT_THRESHOLD=""
RT_CLASS=""
RT_BANK=""
RT_THREADS="${THREADS:-8}"
RT_DEVICE="${DEVICE:-cuda:0}"
RT_CHASSIS="${NANO_CHASSIS:-}"
RT_DEPLOY=0
RT_DRYRUN=0
RT_BAKE_RECALL=""
# Truth-label definition the track-DNN trainer uses. "mtv" (default) trains against matchedAny
# (a track is fake iff it matches no TrackingParticle at all, which is what MTV and the DQM
# plots call a fake) and quotes recall on the narrow, efficiency-selected `matched`. "legacy"
# trains against `matched` itself, which makes real non-efficiency tracks negatives. The
# track-level dumps below always carry matchedAny (NANO_MTV_LABEL=1).
RT_LABEL="${RT_LABEL:-mtv}"
RT_MEM_GUARD_GB="${MEM_GUARD_GB:-300}"

rt_parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --work|--workdir) RT_WORK="${2:?--work needs a directory}"; shift 2 ;;
      --input|--files)  rt_add_input "${2:?--input needs a file list}"; shift 2 ;;
      --sample)         RT_SAMPLE="${2:?--sample needs a name}"; shift 2 ;;
      --events)         RT_EVENTS="${2:?--events needs a number}"; shift 2 ;;
      --dataset)        rt_add_datasets "${2:?--dataset needs a path}"; shift 2 ;;
      --for)            RT_FOR="${2:?--for needs a step name}"; shift 2 ;;
      --threshold)      RT_THRESHOLD="${2:?--threshold needs a value or 'rule'}"; shift 2 ;;
      --class)          RT_CLASS="${2:?--class needs both|forest|mlp}"; shift 2 ;;
      --bank)           RT_BANK="${2:?--bank needs prompt|displaced}"; shift 2 ;;
      --bake-recall)    RT_BAKE_RECALL="${2:?--bake-recall needs a recall, e.g. 0.995}"; shift 2 ;;
      --label)          RT_LABEL="${2:?--label needs mtv|legacy}"; shift 2 ;;
      --threads)        RT_THREADS="${2:?}"; shift 2 ;;
      --device)         RT_DEVICE="${2:?}"; shift 2 ;;
      --chassis)        RT_CHASSIS="${2:?--chassis needs a python module name}"; shift 2 ;;
      --deploy)         RT_DEPLOY=1; shift ;;
      --dry-run)        RT_DRYRUN=1; shift ;;
      --*=*)            set -- "${1%%=*}" "${1#*=}" "${@:2}" ;;
      -h|--help)        rt_usage; exit 0 ;;
      -*)               rt_die "unknown option '$1' (try --help)" ;;
      *)  if [ -z "$RT_STEP" ]; then RT_STEP="$1"; else rt_die "unexpected argument '$1'"; fi; shift ;;
    esac
  done
}

rt_add_datasets() {  # accepts one path, or several separated by commas
  local IFS=, d
  for d in $1; do [ -n "$d" ] && RT_DATASETS+=("$d"); done
}

# --input accepts three forms, so a run over several samples can take its events
# from several places:
#
#   --input /a                                one input for every --sample  (as before)
#   --input ttbarPU=/a,displacedPU=/b,qcdPU=/c   one input per named sample
#   --input /a --input /b --input /c          repeatable, paired positionally
#                                             with the names in --sample
#
# In the named form a chunk starting '<name>=' opens a new sample and every
# following comma-separated chunk without a '<name>=' prefix belongs to it, so a
# sample may still carry a comma-separated file list of its own:
#   --input 'ttbarPU=/a/x.root,/a/y.root,qcdPU=/b'
rt_add_input() {
  local raw="$1" cur="" chunk IFS=,
  RT_INPUT_ORDER+=("$raw")
  # The first --input also seeds RT_INPUT, which is what "can a dataset be
  # produced" and the chassis generation look at.
  [ -n "$RT_INPUT" ] || RT_INPUT="$raw"
  for chunk in $raw; do
    [ -n "$chunk" ] || continue
    if [[ "$chunk" =~ ^[A-Za-z][A-Za-z0-9_.-]*= ]]; then
      cur="${chunk%%=*}"
      RT_INPUT_NAMES+=("$cur")
      RT_INPUT_SPECS+=("${chunk#*=}")
    elif [ -n "$cur" ]; then
      RT_INPUT_SPECS[${#RT_INPUT_SPECS[@]}-1]="${RT_INPUT_SPECS[${#RT_INPUT_SPECS[@]}-1]},$chunk"
    fi
  done
}

# The raw input spec to use for one sample. Named mapping first, then positional
# pairing with the order the samples were given, then the single global input.
rt_input_for() {  # $1 = sample name, $2 = its index in the sample list
  local sample="$1" idx="${2:-0}" i
  for ((i = 0; i < ${#RT_INPUT_NAMES[@]}; i++)); do
    if [ "${RT_INPUT_NAMES[$i]}" = "$sample" ]; then echo "${RT_INPUT_SPECS[$i]}"; return 0; fi
  done
  if [ ${#RT_INPUT_NAMES[@]} -gt 0 ]; then
    # A named mapping was given but this sample is not in it: refuse rather than
    # silently dumping it from somebody else's files.
    rt_die "no --input for sample '$sample' (the named --input mapping covers: ${RT_INPUT_NAMES[*]})"
  fi
  if [ ${#RT_INPUT_ORDER[@]} -gt 1 ]; then
    [ "$idx" -lt ${#RT_INPUT_ORDER[@]} ] ||
      rt_die "sample '$sample' has no --input: ${#RT_INPUT_ORDER[@]} --input option(s) for more samples than that. Give one --input per --sample, or use --input <name>=<files>,..."
    echo "${RT_INPUT_ORDER[$idx]}"; return 0
  fi
  echo "$RT_INPUT"
}

# True when a dataset can be produced at all: some input, or a chassis (which
# carries its own file list).
rt_can_produce() { [ -n "$RT_INPUT" ] || [ -n "$RT_CHASSIS" ]; }

# Alternative step names.
rt_canonical_step() {
  case "$1" in
    nano)   echo dump ;;
    stage1) echo gate ;;
    stage2) echo hp ;;
    train)  echo "" ;;   # ambiguous: the caller must say which model
    *)      echo "$1" ;;
  esac
}

# ---- running ----------------------------------------------------------------
# The PID of the command rt_run is currently running is recorded here so the
# memory watchdog can stop exactly that process tree.
RT_PIDFILE=""

rt_run() {  # echo the command, then run it (unless --dry-run)
  echo "+ $*"
  [ "$RT_DRYRUN" -eq 1 ] && return 0
  if [ -z "$RT_PIDFILE" ]; then "$@"; return $?; fi
  # 0<&0 is an explicit redirection, so the child keeps this shell's stdin
  # instead of the /dev/null a background job gets without job control.
  "$@" 0<&0 &
  local pid=$!
  echo "$pid" >> "$RT_PIDFILE"
  wait "$pid"; local rc=$?
  grep -vx "$pid" "$RT_PIDFILE" > "$RT_PIDFILE.tmp" 2>/dev/null || true
  mv -f "$RT_PIDFILE.tmp" "$RT_PIDFILE" 2>/dev/null || true
  return $rc
}

rt_cmsenv() {
  # shellcheck disable=SC1091
  source /cvmfs/cms.cern.ch/cmsset_default.sh
  eval "$(cd "$RT_SRC" && scramv1 runtime -sh)"
  # The dump configs import their base reconstruction config as a plain python
  # module, so its directory has to be importable. RT_SRC covers a base config
  # kept at the top of the checkout; the working directory covers a generated
  # one.
  export PYTHONPATH="$RT_SRC${RT_WORK:+:$RT_WORK}${PYTHONPATH:+:$PYTHONPATH}"
}

rt_require_no_cmsrun() {
  # RT_ALLOW_CONCURRENT_CMSRUN=1 skips this guard (for nodes with the memory to run both).
  if [ -z "${RT_ALLOW_CONCURRENT_CMSRUN:-}" ] && pgrep -x cmsRun >/dev/null; then
    rt_die "a cmsRun job is running on this node -- training and dumping next to it fights for memory. Wait for it to finish (or set RT_ALLOW_CONCURRENT_CMSRUN=1)."
  fi
}

rt_require_dump_build() {
  if ! grep -Eq '^[[:space:]]*#define[[:space:]]+CA_TRIPLET_DUMP' "$RT_DUMP_MACRO"; then
    echo "ERROR: the per-triplet dump is not compiled in."
    echo "       Uncomment '#define CA_TRIPLET_DUMP' in"
    echo "         $RT_DUMP_MACRO"
    echo "       then rebuild:  scram b code-format && scram b -j"
    echo "       Re-comment it and rebuild again once the dump is done: the production"
    echo "       build must not carry it."
    exit 2
  fi
}

# ---- working directory ------------------------------------------------------
rt_require_work() {
  [ -n "$RT_WORK" ] || rt_die "no working directory: pass --work <dir> (datasets, models and logs are written there)"
  mkdir -p "$RT_WORK"
  RT_WORK="$(cd "$RT_WORK" && pwd)"
  # One file per shell, so two steps running side by side never read each
  # other's children.
  RT_PIDFILE="$RT_WORK/.rt_children.$$"
  : > "$RT_PIDFILE"
  # The trainers write their intermediate model/result files here rather than
  # into the source tree.
  export MODELS_WORKDIR="$RT_WORK"
}

# ---- base reconstruction config (chassis) -----------------------------------
# Resolution order:
#   1. --chassis / NANO_CHASSIS   : an explicit python module name
#   2. --input given              : generate one with make_base_config.sh into
#                                   the working directory (reused afterwards)
#   3. neither                    : leave it unset and let the dump config fall
#                                   back to its own built-in search + sample list
RT_CHASSIS_MODULE=""
rt_resolve_chassis() {
  if [ -n "$RT_CHASSIS" ]; then
    RT_CHASSIS_MODULE="$RT_CHASSIS"
    echo ">>> base reconstruction config: $RT_CHASSIS_MODULE (given; its own process modifiers apply)"
    return
  fi
  if [ -z "$RT_INPUT" ]; then
    RT_CHASSIS_MODULE=""
    echo ">>> base reconstruction config: none given -- the dump config will look for one"
    echo "    next to the checkout. Pass --input <files> to have one generated instead."
    return
  fi
  # One chassis is enough even with per-sample inputs: its --filein list is only
  # a placeholder, because every dump overrides the input files with NANO_FILES /
  # DS_FILES for its own sample. It is generated from the first --input given.
  local out="$RT_WORK/base_reco_cfg.py" first="$RT_INPUT"
  [ ${#RT_INPUT_SPECS[@]} -gt 0 ] && first="${RT_INPUT_SPECS[0]}"
  # Concurrent dumps share this file: generate it under a lock, and reuse a chassis that already
  # exists and parses, so parallel jobs never interleave their cmsDriver writes into one file. The
  # lock serialises the generation; the parse check makes later jobs reuse the first job's file.
  (
    flock -w 600 9 || { echo "!! could not lock $out.lock"; exit 1; }
    if [ -s "$out" ] && python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$out" 2>/dev/null; then
      echo ">>> base reconstruction config: reusing $out (valid, generated by a concurrent dump of this batch)"
    else
      rt_run "$RT_MODELS/make_base_config.sh" --files "$first" --out "$out"
    fi
  ) 9>"$out.lock" || return 1
  RT_CHASSIS_MODULE="base_reco_cfg"
}

# ---- dataset dumps ----------------------------------------------------------
# Track-level dataset: one row per reconstructed track (fit + hit/stub features
# and the matching truth). Feeds the track DNN (quality 'loose') and the final
# high-purity selector (quality 'tight'). arm is prompt | displaced | merged; the
# first two choose which iteration's chain state is set, 'merged' additionally
# turns on the third (merged-collection) row space.
rt_dump_track_nano() {  # $1 = arm, $2 = quality, $3 = sample, $4 = output, $5 = sample index
  local arm=$1 quality=$2 sample=$3 out=$4 idx=${5:-0} spec
  local -a env=(NANO_SAMPLE="$sample" NANO_MINQUALITY="$quality" NANO_THREADS="$RT_THREADS"
                NANO_NEVT="${RT_EVENTS:-1000}" NANO_OUT="$out")
  # The MTV-true label columns on the per-arm truth tables. Independent of arm and of quality,
  # so the track-DNN dataset (quality loose) carries them exactly as the high-purity one (quality
  # tight) does: both trainers target matchedAny and quote recall on matched.
  env+=(NANO_MTV_LABEL=1)
  # arm = merged: the same tight dump, plus the TrkMerged* row space over
  # hltPhase2PixelTracksSoAMerger -- the population the FINAL high-purity selector decides on. It is
  # produced by BOTH iterations, so BOTH arms' combinatorics buffers are enlarged: a truncation in
  # either one would silently remove merged tracks and bias exactly the trained population. Nothing
  # else about the chain state changes, so the per-arm tables in the same file stay usable. The
  # merged selector also reads the seven pixel-cluster columns, which NANO_CLUSTER=1 adds.
  if [ "$arm" = merged ]; then
    env+=(NANO_MERGED=1 NANO_CLUSTER=1
          NANO_PROMPT_CAPS=big NANO_PROMPT_CAPS_FACTOR="${RT_PROMPT_CAPS_FACTOR:-4}"
          NANO_DISP_CAPS=big NANO_DISP_CAPS_FACTOR="${RT_DISP_CAPS_FACTOR:-4}")
  fi
  [ -n "$RT_CHASSIS_MODULE" ] && env+=(NANO_CHASSIS="$RT_CHASSIS_MODULE")
  spec=$(rt_input_for "$sample" "$idx")
  [ -n "$spec" ] && env+=(NANO_FILES="$(rt_file_list "$spec")")
  # The track DNN has to be trained on the tracks it will decide on, i.e. the
  # population BEFORE the currently deployed bank filtered it. So the dump for
  # the 'loose' dataset switches that selector off; every other dump inherits
  # whatever the configuration carries.
  #
  # Both arms also enlarge the combinatorics buffers for the loose dump:
  # with the promotion selector off, many more tracks reach the containers than
  # in the production configuration, and an overflow drops them silently -- which
  # would bias exactly the population this model is trained on. The displaced arm
  # needs it for the same reason as the prompt one, so it is set for both.
  if [ "$quality" = loose ]; then
    if [ "$arm" = prompt ]; then env+=(NANO_PROMPT_TRACKDNN=0 NANO_PROMPT_CAPS=big)
    else env+=(NANO_DISP_TRACKDNN=0 NANO_DISP_CAPS=big); fi
  fi
  # The displaced arm is the one being trained in the displaced chain: enlarge ITS buffers for the
  # tight (HP) dump as well, so no training item is lost to container truncation. The prompt arm
  # (frozen upstream) keeps its production caps. RT_DISP_CAPS_FACTOR (default 4) sets the factor.
  if [ "$arm" = displaced ]; then env+=(NANO_DISP_CAPS=big NANO_DISP_CAPS_FACTOR="${RT_DISP_CAPS_FACTOR:-4}"); fi
  rt_run env "${env[@]}" cmsRun "$RT_MODELS/trackNano_config.py" 2>&1 | tee "$RT_WORK/$(basename "${out%.root}").log"
  echo ">>> the chain state this dataset is valid for:"
  grep -E "^\[trackNano CHAIN-STATE\]" "$RT_WORK/$(basename "${out%.root}").log" ||
    echo "    WARNING: no chain-state line in the log -- check it before using this dataset."
}

# Triplet dataset: one row per triplet the cellular automaton built, with the
# truth label. Feeds the in-kernel triplet gate. Needs the dump build.
# The default event count covers the scoring window retrain_triplet_gate.sh uses
# by default -- events [TEST_SKIP, TEST_SKIP+TEST_EV) = [1400, 2000) -- because
# the trainer's windowing is per file. A smaller dump is fine: the worker derives
# a proportional window from the event count it actually finds.
RT_TRIPLET_NEVT_DEFAULT=2000
rt_dump_triplet_nano() {  # $1 = arm, $2 = sample, $3 = output, $4 = sample index
  local arm=$1 sample=$2 out=$3 idx=${4:-0} spec
  local iter=disp; [ "$arm" = prompt ] && iter=prompt
  local -a env=(DS_NTUPLE_ITER="$iter" DS_SAMPLE="$sample"
                DS_NEVT="${RT_EVENTS:-$RT_TRIPLET_NEVT_DEFAULT}"
                DS_NTUPLE_THREADS="$RT_THREADS" DS_NTUPLE_OUT="$out")
  [ -n "$RT_CHASSIS_MODULE" ] && env+=(DS_CHASSIS="$RT_CHASSIS_MODULE")
  spec=$(rt_input_for "$sample" "$idx")
  [ -n "$spec" ] && env+=(DS_FILES="$(rt_file_list "$spec")")
  rt_run env "${env[@]}" cmsRun "$RT_MODELS/triplet_dump_cfg.py" 2>&1 | tee "$RT_WORK/$(basename "${out%.root}").log"
}

# Turn --input into the comma-separated list the dump configs read. It may be a
# comma-separated list of files, a directory of .root files, or a text file with
# one .root path per line.
rt_file_list() {  # $1 = input spec (default: the first --input given)
  local spec="${1:-$RT_INPUT}" out=""
  if [ -d "$spec" ]; then
    out=$(ls -1 "$spec"/*.root 2>/dev/null | paste -sd,)
    [ -n "$out" ] || rt_die "no .root files in the directory '$spec'"
  elif [ -f "$spec" ] && [[ "$spec" != *.root ]]; then
    out=$(grep -vE '^[[:space:]]*(#|$)' "$spec" | grep -E '\.root$' | paste -sd,)
    [ -n "$out" ] || rt_die "'$spec' is not a .root file and holds no .root paths -- --input takes event files, a directory of them, or a list of their paths"
  else
    out="$spec"
  fi
  echo "$out"
}

# ---- how many events a produced dataset actually holds ----------------------
# The triplet trainer windows PER FILE (events [skip, skip+max) of each dataset),
# so the scoring window has to fit inside the smallest dataset of the run. This
# reports that number; it prints nothing if uproot is unavailable or a file
# cannot be read, and callers fall back to their documented defaults.
rt_min_events() {  # $@ = dataset files (a trailing ':label' is tolerated)
  python3 - "$@" <<'PYEOF' 2>/dev/null
import sys
try:
    import uproot
except Exception:
    sys.exit(0)
n = None
for a in sys.argv[1:]:
    p = a.rsplit(":", 1)[0] if (":" in a and not a.endswith(".root")) else a
    try:
        e = uproot.open(p)["Events"].num_entries
    except Exception:
        sys.exit(0)
    n = e if n is None else min(n, e)
if n is not None:
    print(int(n))
PYEOF
}

# ---- what the deployed configuration currently carries ----------------------
# $1 = file, $2 = parameter, $3 = module label (optional) -> the value, or empty.
#
# The third argument matters: a cfi may declare SEVERAL producers (the prompt HP selector file
# declares the upstream Torch baseline scoreThreshold=0.4 AND the phase2CAStubs forest
# scoreThreshold=0.06 the stub chain actually runs). Reading the first match reports the wrong
# module's working point. With a module label the search is confined to that producer's block
# (from '<label> = cms.EDProducer(' down to the line that closes it).
rt_cfi_double() {
  local file=$1 par=$2 mod=${3:-} body
  if [ -n "$mod" ]; then
    body=$(awk -v m="$mod" '
      $0 ~ "^_?" m "[[:space:]]*=[[:space:]]*cms\\.EDProducer\\(" { inb = 1 }
      inb { print }
      inb && /^\)/ { exit }' "$file" 2>/dev/null)
    [ -n "$body" ] || body=$(cat "$file" 2>/dev/null)   # label not found: fall back to the file
  else
    body=$(cat "$file" 2>/dev/null)
  fi
  printf '%s\n' "$body" |
    grep -Eo "^[[:space:]]*$par[[:space:]]*=[[:space:]]*cms\.double\([0-9eE.+-]+\)" |
    head -1 | grep -Eo '\([0-9eE.+-]+\)' | tr -d '()'
}

# The label of the producer the stub chain actually runs in a high-purity
# selector cfi: the LAST PixelTrackForestHighPuritySelector@alpaka declared in
# it (the prompt file declares the Torch baseline first and swaps this one in
# under phase2CAStubs; the displaced file declares only this one). Empty if the
# file has none, in which case rt_cfi_double falls back to the whole file.
rt_hp_module() {  # $1 = high-purity selector cfi
  grep -Eo "^_?[A-Za-z0-9_]+[[:space:]]*=[[:space:]]*cms\.EDProducer\('PixelTrackForestHighPuritySelector@alpaka'" \
    "$1" 2>/dev/null | tail -1 | sed -E "s/^_?([A-Za-z0-9_]+).*/\1/"
}

rt_baked_threshold() {  # $1 = baked weights header -> its built-in threshold
  grep -Eo 'kDefaultThreshold[[:space:]]*=[[:space:]]*[0-9eE.+-]+' "$1" 2>/dev/null |
    head -1 | grep -Eo '[0-9]\.[0-9eE.+-]+' | head -1
}

rt_deployed_model() {  # $1 = high-purity selector config -> the model file it loads
  grep -Eo "[A-Za-z0-9_.-]+\.bin" "$1" 2>/dev/null | head -1
}

rt_model_stem() {  # strip a trailing _YYYYMMDD date and the extension
  local b=${1##*/}; b=${b%.bin}; echo "$b" | sed -E 's/_[0-9]{8}$//'
}

# ---- the final-selector forests ----------------------------------------------
# The three selectors (prompt, displaced, merged) are trained by the same script,
# train_merged_forest.py; what differs per collection is the cache, the recipe
# flags and the configuration file. The shared part lives here.
RT_TRAINER="$RT_MODELS/train_merged_forest.py"
RT_PROFILER="$RT_MODELS/make_profile.py"

# The working-point rule of a retrain (train_merged_forest.py --wp-rule):
#   uniform  (default) recall >= --recall in every pT / |eta| / |dxyBS| bin
#   global   recall = --recall over all true tracks
#   profile  match or exceed, bin by bin, the model the configuration loads today,
#            measured on this run's own cache by make_profile.py
RT_WP_RULE="${WP_RULE:-uniform}"

# Sets RT_WP_ARGS, the trainer's working-point options for the chosen rule. For
# 'profile' it first measures the deployed model on the cache.
RT_WP_ARGS=()
rt_wp_args() {  # $1 = arm, $2 = cache, $3 = deployed .bin, $4 = its threshold, $5 = profile out, $6.. = make_profile.py feature options
  RT_WP_ARGS=()
  case "$RT_WP_RULE" in
    uniform|global) RT_WP_ARGS=(--wp-rule "$RT_WP_RULE") ;;
    profile)
      local arm=$1 cache=$2 model=$3 thr=$4 out=$5; shift 5
      [ -n "$model" ] && [ -f "$model" ] || rt_die "WP_RULE=profile needs the deployed model file (found '${model:-none}')"
      [ -n "$thr" ] || rt_die "WP_RULE=profile: no scoreThreshold found in the configuration"
      echo ">>> reference profile: the deployed $(basename "$model") at $thr, measured on this cache"
      rt_run python3 "$RT_PROFILER" --cache "$cache" --model "$model" --threshold "$thr" \
        --arm "$arm" --split test --name "deployed_$(rt_model_stem "$model")" --out "$out" "$@"
      RT_WP_ARGS=(--wp-rule profile --wp-profile "$out" --wp-margin 0) ;;
    *) rt_die "WP_RULE must be uniform, global or profile (got '$RT_WP_RULE')" ;;
  esac
}

# What the trainer produced, read from its result.json (or the names it will use,
# under --dry-run).
rt_forest_bin() {  # $1 = trainer --out dir, $2 = arm, $3 = width, $4 = tag, $5 = date
  if [ "$RT_DRYRUN" -eq 1 ] || [ ! -f "$1/result.json" ]; then echo "$1/${2}_tree${3}_${4}_${5}.bin"; return 0; fi
  python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['export']['bin'])" "$1/result.json"
}
rt_forest_wp() {  # $1 = trainer --out dir -> the offline working point, as a sentence
  if [ "$RT_DRYRUN" -eq 1 ] || [ ! -f "$1/result.json" ]; then echo "<the working point in $1/result.json>"; return 0; fi
  python3 -c "import json,sys;w=json.load(open(sys.argv[1]))['working_point'];print('%.6f offline (rule %s, recall %.4f, fake rejection %.4f)'%(w['threshold'],w['mode'],w['recall_achieved'],w['rejection']))" "$1/result.json"
}
rt_r_forest_deploy() {  # $1 = staged .bin, $2 = the file the configuration loads today
  rt_r_deploy "cp $1 $RT_DATA/" \
              "no rebuild needed -- the selector reads the file at run time" \
              "(the file the configuration loads today is ${2:-unknown}; keeping the date in the" \
              " name is what makes a model swap visible in the configuration diff)"
}

# ---- memory watchdog --------------------------------------------------------
# The cgroup 'anon' figure is the one to watch: most of `memory.current` on this
# kind of node is reclaimable file cache and a guard on it trips for no reason.
# Stop a process and everything it started, by PID. Never by name: 'pkill -f
# <script>.py' also matches the wrapper shell, so a trip could take down an
# unrelated job of the same user.
rt_kill_tree() {  # $1 = pid, $2 = signal (default TERM)
  local pid=$1 sig=${2:-TERM} c
  kill -0 "$pid" 2>/dev/null || return 0
  for c in $(pgrep -P "$pid" 2>/dev/null); do rt_kill_tree "$c" "$sig"; done
  kill -"$sig" "$pid" 2>/dev/null || true
}

rt_start_guard() {  # $1 = log file; any further words are only noted in the log
  local log=$1; shift
  local note="$*"
  if [ "$RT_DRYRUN" -eq 1 ]; then ( : ) & echo $!; return; fi
  local pidfile="$RT_PIDFILE"
  ( while true; do
      anon=$(( $(grep -m1 '^anon ' /sys/fs/cgroup/memory.stat | awk '{print $2}') / 1073741824 ))
      echo "$(date +%H:%M:%S) anon=${anon}G/${RT_MEM_GUARD_GB}G${note:+ ($note)}" >> "$log"
      if [ "$anon" -gt "$RT_MEM_GUARD_GB" ]; then
        echo "$(date +%H:%M:%S) memory guard tripped" >> "$log"
        if [ -n "$pidfile" ] && [ -s "$pidfile" ]; then
          while read -r pid; do
            [ -n "$pid" ] || continue
            echo "$(date +%H:%M:%S) stopping pid $pid and its children" >> "$log"
            rt_kill_tree "$pid" TERM
          done < "$pidfile"
          sleep 10
          while read -r pid; do
            [ -n "$pid" ] || continue
            rt_kill_tree "$pid" KILL
          done < "$pidfile"
        else
          echo "$(date +%H:%M:%S) no child recorded -- nothing stopped" >> "$log"
        fi
        break
      fi
      sleep 60
    done ) >/dev/null 2>&1 & echo $!
  # (stdout/stderr of the guard are detached: callers capture this function with $(...), and a
  #  background child that inherits the captured pipe would keep the substitution blocked forever)
}

# ---- end-of-step report -----------------------------------------------------
# Every step that produces something ends with the same three blocks: what was
# written, where it has to go, and which configuration values have to be
# revisited.
rt_report() {  # $1 = title; then PRODUCED/DEPLOY/UPDATE lines via rt_r_* helpers
  echo
  echo "============================================================================"
  echo " $1"
  echo "============================================================================"
}
rt_r_produced() { echo " PRODUCED"; local l; for l in "$@"; do echo "   $l"; done; echo; }
rt_r_deploy()   { echo " DEPLOY"; local l; for l in "$@"; do echo "   $l"; done; echo; }
rt_r_update()   { echo " UPDATE"; local l; for l in "$@"; do echo "   $l"; done; echo; }
rt_r_next()     { echo " NEXT"; local l; for l in "$@"; do echo "   $l"; done; echo; }
