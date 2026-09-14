#!/bin/bash
# =============================================================================
# retrain_track_dnn.sh -- train and bake the in-kernel track DNN (the
# loose -> tight promotion inside the track classification kernel).
# =============================================================================
# Worker behind `retrain_prompt.sh gate` and `retrain_displaced.sh gate`, so
# both iterations get the same method. It can also be run directly:
#
#   ./retrain_track_dnn.sh --bank prompt|displaced --work DIR \
#                          --dataset <trackNano_*_loose.root> [--dataset ...]
#
# THE MODEL. A 12-feature network on fit quality plus hit/stub geometry, replacing a bare chi2
# cut on the loose->tight promotion. On the prompt iteration it is where most of the fake
# rejection happens, which is why the selector after it can run looser than its own offline
# rule suggests.
#
# THE DATASET. It must hold the population this model decides on, before any such decision:
# quality 'loose', the deployed track DNN switched OFF at dump time, the per-triplet gate at its
# deployed threshold. `retrain_*.sh dump --for gate` sets all three and prints the state it used.
# Re-run this step after any change to the track fit: the features are fit outputs, so a bank
# trained on an older fit is quietly mismatched (compare_track_dnn_banks.py makes it visible).
#
# THRESHOLD. Stating the rule as a recall is legitimate here: one decision per track means track
# survival == track recall (the per-triplet gate compounds, see retrain_triplet_gate.sh). By
# default the value baked is the one track_dnn_working_point.py derives: the largest threshold at
# which the new bank keeps at least the outgoing bank's fraction of matched tracks in EVERY pT,
# |eta| and |dxy| bin of this run's own test split. So a retrain with no options moves the model
# and moves the threshold with it, to the point where nothing the old bank kept is given up
# anywhere. --threshold <x> (or THRESHOLD=<x>) bakes an explicit value instead, --threshold rule
# the trainer's own recall point. The baked value is a fallback: the producer's trackDNNThreshold
# overrides it.
#
# LABEL. The trainer targets the MTV-true label (matchedAny: matched to ANY TrackingParticle,
# MTV's "not a fake") and quotes recall on the efficiency-selected `matched`; the fake rejection
# printed next to it is against the MTV class, the two axes of the DQM plots. --label legacy
# targets `matched` itself, so real tracks failing an efficiency cut count as fakes. The MTV
# label needs a dump carrying matchedAny (NANO_MTV_LABEL=1, set by retrain_common.sh).
#
# RECALL POINT. train_disp_nano.py reports two working points, 99.0 % and 99.5 % displacement-
# weighted recall. The bake reads 99.0 % by default; --bake-recall 0.995 keeps another half
# percent of displaced real tracks at the cost of free fake rejection. It implies --threshold
# rule; an explicit --threshold <x> still wins.
#
# OPTIONS
#   --bank prompt|displaced   which iteration's weights to train    (required)
#   --work DIR                models and logs                       (required)
#   --dataset FILE            training dataset (repeatable)
#   --threshold X|rule        threshold to bake (default: the per-bin working point,
#                             see THRESHOLD above; 'rule' = the trainer's recall point)
#   --bake-recall R           with --threshold rule, take the trainer's point at this
#                             displacement-weighted recall: 0.99 (default) or 0.995
#   --label mtv|legacy        truth-label definition (default mtv, see above)
#   --device DEV              torch device                          (default cuda:0)
#   --dry-run                 print the commands without running them
#
# ENVIRONMENT
#   NAME            artifact tag                    (default <bank>_stage1_12f)
#   THRESHOLD       an explicit threshold to bake, as --threshold
#   MEM_GUARD_GB    stop if the cgroup anonymous memory exceeds this (default 300)
# =============================================================================
set -uo pipefail
# shellcheck source=retrain_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/retrain_common.sh"

rt_usage() { sed -n '/^# OPTIONS/,/^# ====/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//;$d'; }

rt_parse_args "$@"
BANK=${RT_BANK:-${RT_STEP:-}}
[ -n "$BANK" ] || rt_die "--bank is required: prompt or displaced (it selects the weight bank this run overwrites)"
case "$BANK" in prompt|displaced) ;; *) rt_die "--bank must be prompt or displaced" ;; esac
[ ${#RT_DATASETS[@]} -gt 0 ] || rt_die "no dataset: pass --dataset <trackNano_*_loose.root>"
rt_require_work

NAME=${NAME:-${BANK}_stage1_12f}
HEADER="$RT_PLUGINS/CATrackDNNWeights_${BANK}.h"
CA_CFI=$(rt_ca_cfi "$BANK")

# The threshold the chain runs today: the producer value if it sets one, else
# the value baked into the header this run replaces.
CURRENT_THR=$(rt_cfi_double "$CA_CFI" trackDNNThreshold)
CURRENT_SRC="the producer configuration"
if [ -z "$CURRENT_THR" ]; then
  CURRENT_THR=$(rt_baked_threshold "$HEADER")
  CURRENT_SRC="the header being replaced"
fi
# Empty means "derive the per-bin working point below"; an explicit --threshold
# (or THRESHOLD in the environment) always wins.
BAKE_THR=${RT_THRESHOLD:-}

# --bake-recall asks for the trainer's own recall point, so it selects the rule
# branch unless an explicit numeric threshold was given (which always wins).
BAKE_RECALL=${RT_BAKE_RECALL:-}
if [ -n "$BAKE_RECALL" ]; then
  if [ -n "$BAKE_THR" ] && [ "$BAKE_THR" != rule ]; then
    echo "NOTE: --threshold $BAKE_THR is explicit, so --bake-recall $BAKE_RECALL is not used."
    BAKE_RECALL=""
  else
    BAKE_THR=rule
  fi
fi

rt_require_no_cmsrun
GUARD=$(rt_start_guard "$RT_WORK/track_dnn_ram.log" "track DNN, bank=$BANK")
trap 'kill $GUARD 2>/dev/null' EXIT

echo "=== [1/4] train: bank=$BANK, tag=$NAME, label=$RT_LABEL, ${#RT_DATASETS[@]} dataset(s) ==="
# --bank selects both the feature table and the objective: the prompt bank uses
# plain cross-entropy (a near-beamline population has no displacement axis), the
# displaced bank weights by displacement. Both are the trainer's own per-bank
# defaults; leave them alone.
rt_run python3 "$RT_MODELS/train_disp_nano.py" train "${RT_DATASETS[@]}" \
  --bank "$BANK" --device "$RT_DEVICE" --name "$NAME" --label "$RT_LABEL" \
  2>&1 | tee "$RT_WORK/track_dnn_train_${BANK}.log"
[ "${PIPESTATUS[0]}" -ne 0 ] && { echo "training failed"; exit 1; }

echo "=== [2/4] compare the new model against the bank in use, at equal track recall ==="
# This scores the HEADER, i.e. the thing that actually runs on the device, by
# reproducing its forward pass. Everything between the trained model and the
# header (weight layout, standardisation arrays, text round-trip) is otherwise
# only exercised at build time, so a baking mistake or a bank left over from an
# older fit shows up here.
rt_run python3 "$RT_MODELS/compare_track_dnn_banks.py" "${RT_DATASETS[@]}" --bank "$BANK" \
  --header "$HEADER" --pt "$RT_WORK/model_${NAME}.pt" \
  2>&1 | tee "$RT_WORK/track_dnn_comparison_${BANK}.txt"
[ "${PIPESTATUS[0]}" -ne 0 ] && { echo "the comparison against the bank in use failed; $HEADER is untouched"; exit 1; }

# The working point of the bank being replaced, bin by bin, measured on this run's
# own test split. The header is still the outgoing one here -- the bake is the next
# step -- which is exactly the comparison the rule asks for.
WP_LOG="$RT_WORK/track_dnn_working_point_${BANK}.txt"
DERIVE_WP=0; [ -z "$BAKE_THR" ] && DERIVE_WP=1
if [ "$DERIVE_WP" = 1 ]; then
  echo "=== [3/4] working point: the largest threshold holding the outgoing bank's recall in every bin ==="
  rt_run python3 "$RT_MODELS/track_dnn_working_point.py" "${RT_DATASETS[@]}" --bank "$BANK" \
    --old-header "$HEADER" --pt "$RT_WORK/model_${NAME}.pt" \
    2>&1 | tee "$WP_LOG"
  [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "the working-point scan failed; $HEADER is untouched"; exit 1; }
  if [ "$RT_DRYRUN" -eq 0 ]; then
    BAKE_THR=$(sed -n 's/^CHOSEN per-bin threshold \([0-9.eE+-]*\) .*/\1/p' "$WP_LOG" | tail -1)
    [ -n "$BAKE_THR" ] ||
      { echo "no threshold in $WP_LOG (expected a 'CHOSEN per-bin threshold' line); $HEADER is untouched"; exit 1; }
    echo "    working point $BAKE_THR   (${CURRENT_THR:-unset} today, from $CURRENT_SRC)"
  fi
else
  echo "=== [3/4] working point: skipped, the threshold was given ($BAKE_THR) ==="
fi

echo "=== [4/4] bake -> $HEADER ==="
# Keep a copy of the header being replaced, in the working directory rather than
# next to the source file, so the source tree gains nothing but the new header.
BACKUP="$RT_WORK/$(basename "$HEADER").replaced"
if [ "$RT_DRYRUN" -eq 0 ]; then cp -p "$HEADER" "$BACKUP" 2>/dev/null || true; fi

# A failed bake can leave a half-written header behind, and the report below would
# still claim success. Put the previous bank back and stop.
bake_failed() {
  echo "ERROR: the bake failed" >&2
  if [ "$RT_DRYRUN" -eq 0 ] && [ -f "$BACKUP" ]; then
    cp -p "$BACKUP" "$HEADER" && echo "       $HEADER restored from $BACKUP" >&2
  fi
  exit 1
}
if [ "$DERIVE_WP" = 1 ] && [ "$RT_DRYRUN" -eq 1 ]; then
  echo "    baking the per-bin working point step [3/4] prints"
  rt_run python3 "$RT_MODELS/train_disp_nano.py" bake --bank "$BANK" --name "$NAME" \
    --threshold "<the per-bin working point>" --out "$HEADER"
elif [ -z "$BAKE_THR" ] || [ "$BAKE_THR" = rule ]; then
  echo "    baking the trainer's own point (from result_${NAME}.json)" \
       "at the ${BAKE_RECALL:-0.99} displacement-weighted recall"
  BR=(); [ -n "$BAKE_RECALL" ] && BR=(--bake-recall "$BAKE_RECALL")
  rt_run python3 "$RT_MODELS/train_disp_nano.py" bake --bank "$BANK" --name "$NAME" \
    "${BR[@]}" --out "$HEADER" || bake_failed
else
  echo "    baking $BAKE_THR"
  rt_run python3 "$RT_MODELS/train_disp_nano.py" bake --bank "$BANK" --name "$NAME" \
    --threshold "$BAKE_THR" --out "$HEADER" || bake_failed
fi

if [ "$RT_DRYRUN" -eq 0 ]; then
  echo "=== baked header ==="
  grep -E "kNFeat|kNH1|kNH2|kDefaultThreshold" "$HEADER" | head -6
fi

rt_report "Track DNN retrained ($BANK iteration)"
rt_r_produced "$HEADER   (written in place)" \
              "the header it replaced: $BACKUP" \
              "comparison against the bank in use: $RT_WORK/track_dnn_comparison_${BANK}.txt" \
              "per-bin working point: $WP_LOG"
rt_r_deploy "the weights are compiled into the kernel, so rebuild:" \
            "  scram b code-format && scram b -j" \
            "(never rebuild while a cmsRun job is running)" \
            "code-format reformats the header a bake just wrote; its numbers do not change," \
            "so compare banks with compare_track_dnn_banks.py rather than by checksum."
rt_r_update "$CA_CFI" \
            "  trackDNNThreshold  -- the baked value (${BAKE_THR:-the per-bin working point}) is what runs when this is unset;" \
            "                        ${CURRENT_THR:-<unset>} today, from $CURRENT_SRC"
rt_r_next "the next step (the final high-purity selector) trains on tracks this model" \
          "promoted, so rebuild first, then produce its dataset again:" \
          "  scram b code-format && scram b -j" \
          "  ./retrain_${BANK}.sh hp --work $RT_WORK --input <event files>"
