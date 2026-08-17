#!/bin/bash
# =============================================================================
# retrain_triplet_gate.sh -- train and bake the in-kernel per-triplet gate.
# =============================================================================
# Worker behind `retrain_prompt.sh triplet` and `retrain_displaced.sh triplet`,
# so both iterations get the same method. It can also be run directly:
#
#   ./retrain_triplet_gate.sh --bank prompt|displaced --work DIR \
#                             --dataset <tripletNano.root>[:label] [--dataset ...]
#
# WHAT IT DOES: read the dataset (chunked, memory guarded) -> train the small network that
# scores every triplet the CA builds -> score on an event range DISJOINT from training, broken
# down by displacement, pT, eta, stub count -> pick a threshold -> write CATripletDNNWeights_<bank>.h.
#
# THE THRESHOLD RULE. A track leaves ~10 real triplets behind, so "keep 99 % of real triplets"
# keeps only ~0.99^10 = 90 % of the tracks. The default rule therefore works on per-track
# survival in the worst (displacement, momentum) group, not on a single global triplet recall.
# THRESHOLD_RULE=global selects the global-recall rule for comparison (it also prints what the
# default rule would have chosen).
#
# WHICH NUMBER GETS BAKED. By default the threshold the chain runs today (the producer config
# value if it sets one, else the value in the header being replaced), so a retrain with no
# options replaces the model and changes nothing else. --threshold rule bakes the derived value;
# --threshold <x> an explicit one. The baked value is a fallback: the producer's
# tripletDNNThreshold overrides it, and that value belongs to an in-situ scan.
#
# OPTIONS
#   --bank prompt|displaced   which iteration's weights to train    (required)
#   --work DIR                datasets, models, plots and logs      (required)
#   --dataset FILE[:LABEL]    training dataset; repeatable. The label appears in
#                             the per-sample breakdowns (default: from the name)
#   --threshold X|rule        threshold to bake (default: what the chain runs)
#   --device DEV              torch device                          (default cuda:0)
#   --dry-run                 print the commands without running them
#
# ENVIRONMENT (tuning; the defaults are the ones in use)
#   TRAIN_EV        events [0,TRAIN_EV) used for training      (default 1000)
#   TEST_SKIP/TEST_EV  the disjoint scoring range              (default 1400/600)
#                   The trainer windows PER FILE, so a scoring range past the end of
#                   a dataset shrinks or vanishes. When none of the three is set, the
#                   script counts the events in the smallest dataset and, below 2000,
#                   keeps the 50 % / 20 % gap / 30 % proportions inside what is there
#                   (train [0,0.5N), score [0.7N,N)). Setting any of the three by hand
#                   disables the derivation for that one.
#   VARIANT         training recipe                            (default bal_up05)
#   EPOCHS          maximum epochs                             (default 40)
#   WORKERS         ingestion workers (default: sized from cores and memory)
#   THRESHOLD_RULE  worstgroup-survival (default) | global
#   SURVIVAL_FLOOR  per-track survival the rule must hold      (default 0.90)
#   REF_SURVIVAL    json of a reference model to pin each group against
#   MEM_GUARD_GB    stop if the cgroup anonymous memory exceeds this (default 300)
#   EXTRA           extra flags passed to the trainer
# =============================================================================
set -uo pipefail
# shellcheck source=retrain_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/retrain_common.sh"

rt_usage() { sed -n '/^# OPTIONS/,/^# ====/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//;$d'; }

rt_parse_args "$@"
BANK=${RT_BANK:-${RT_STEP:-}}
[ -n "$BANK" ] || rt_die "--bank is required: prompt or displaced (it selects the weight bank this run overwrites)"
case "$BANK" in prompt|displaced) ;; *) rt_die "--bank must be prompt or displaced" ;; esac
[ ${#RT_DATASETS[@]} -gt 0 ] || rt_die "no dataset: pass --dataset <tripletNano.root>[:label]"
rt_require_work

# Remember whether the caller pinned a window before the defaults are applied.
_TRAIN_EV_SET=${TRAIN_EV+1}; _TEST_SKIP_SET=${TEST_SKIP+1}; _TEST_EV_SET=${TEST_EV+1}
TRAIN_EV=${TRAIN_EV:-1000}
TEST_SKIP=${TEST_SKIP:-1400}
TEST_EV=${TEST_EV:-600}
VARIANT=${VARIANT:-bal_up05}
EPOCHS=${EPOCHS:-40}
WORKERS=${WORKERS:-}
THRESHOLD_RULE=${THRESHOLD_RULE:-worstgroup-survival}
SURVIVAL_FLOOR=${SURVIVAL_FLOOR:-0.90}
REF_SURVIVAL=${REF_SURVIVAL:-}
EXTRA=${EXTRA:-}
OUTDIR=${OUTDIR:-$RT_WORK/plots_tripletDNN}
LOGDIR=${LOGDIR:-$OUTDIR}
HEADER="$RT_PLUGINS/CATripletDNNWeights_${BANK}.h"
CA_CFI=$(rt_ca_cfi "$BANK")

# Dataset specs: '<file>:<label>'. A bare path gets its label from the file name
# (tripletNano_<label>.root), so the per-sample breakdowns stay readable.
SPECS=()
for d in "${RT_DATASETS[@]}"; do
  case "$d" in
    *:*) SPECS+=("$d") ;;
    *)   b=${d##*/}; b=${b%.root}; b=${b#tripletNano_}; SPECS+=("$d:${b:-sample}") ;;
  esac
done

# Fit the training and scoring windows inside the events that were actually
# dumped. The two windows stay disjoint and keep the default proportions.
NEVT_AVAIL=$(rt_min_events "${SPECS[@]}")
if [ -n "$NEVT_AVAIL" ]; then
  echo "=== the smallest dataset of this run holds $NEVT_AVAIL events ==="
  if [ "$NEVT_AVAIL" -lt $((TEST_SKIP + TEST_EV)) ]; then
    if [ -n "$_TRAIN_EV_SET$_TEST_SKIP_SET$_TEST_EV_SET" ]; then
      echo "    WARNING: the scoring window [$TEST_SKIP,$((TEST_SKIP + TEST_EV))) reaches past that."
      echo "             The trainer windows per file, so it will be truncated to"
      echo "             [$TEST_SKIP,$NEVT_AVAIL) -- empty if that is not a range."
      echo "             Dump more events, or leave TRAIN_EV/TEST_SKIP/TEST_EV unset"
      echo "             and they are derived from the dataset."
    else
      TRAIN_EV=$((NEVT_AVAIL * 50 / 100))
      TEST_SKIP=$((NEVT_AVAIL * 70 / 100))
      TEST_EV=$((NEVT_AVAIL - TEST_SKIP))
      echo "    the default window does not fit, so it is scaled to the dataset:"
      echo "    train [0,$TRAIN_EV)  score [$TEST_SKIP,$((TEST_SKIP + TEST_EV)))"
      [ "$TEST_EV" -gt 0 ] ||
        rt_die "only $NEVT_AVAIL event(s) in the dataset -- too few to score on. Dump at least a few hundred (the dump default is $RT_TRIPLET_NEVT_DEFAULT)."
    fi
  fi
fi

# The threshold the chain runs today: the producer value if it sets one, else
# the value baked into the header this run replaces.
CURRENT_THR=$(rt_cfi_double "$CA_CFI" tripletDNNThreshold)
CURRENT_SRC="the producer configuration"
if [ -z "$CURRENT_THR" ]; then
  CURRENT_THR=$(rt_baked_threshold "$HEADER")
  CURRENT_SRC="the header being replaced (the producer sets no threshold, so the baked one is used)"
fi
BAKE_THR=${RT_THRESHOLD:-$CURRENT_THR}

mkdir -p "$OUTDIR" "$LOGDIR"
W=""; [ -n "$WORKERS" ] && W="--workers $WORKERS"
RS=""; [ -n "$REF_SURVIVAL" ] && RS="--ref-survival-json $REF_SURVIVAL"
BT=""
if [ -n "$BAKE_THR" ] && [ "$BAKE_THR" != rule ]; then
  BT="--bake-thr $BAKE_THR"
  cat <<BANNER
============================================================================
 The header will be baked at threshold $BAKE_THR, taken from
 $CURRENT_SRC.
 The rule this run applies ($THRESHOLD_RULE) is still evaluated and printed
 below. If the two disagree that is information, not an error: the deployed
 value comes from a scan of the full reconstruction, on axes an offline scorer
 cannot see. Pass --threshold rule to bake the derived number instead.
============================================================================
BANNER
fi

rt_require_no_cmsrun

# Belt and braces on top of the trainer's own in-process guard.
GUARD=$(rt_start_guard "$LOGDIR/retrain_ram.log" "triplet gate, bank=$BANK")
trap 'kill $GUARD 2>/dev/null' EXIT

export TRIPLET_DNN_OUTDIR="$OUTDIR"   # plots and per-run artifacts stay out of the source tree

echo "=== [1/2] train: recipe=$VARIANT, events [0,$TRAIN_EV), bank=$BANK ==="
rt_run python3 "$RT_MODELS/train_triplet_dnn.py" train "${SPECS[@]}" \
  --bank "$BANK" --device "$RT_DEVICE" --variant "$VARIANT" --max-events "$TRAIN_EV" --chunk-events 10 \
  --max-epochs "$EPOCHS" --patience 10 --val-frac 0.15 \
  --weight-floor 1.0 --weight-cap 30 --weight-normalize bulk \
  --survival-floor "$SURVIVAL_FLOOR" --mem-guard-gb 260 --mem-log "$LOGDIR/retrain_ram.log" \
  $W $EXTRA 2>&1 | tee "$LOGDIR/retrain_train_${VARIANT}.log"
[ "${PIPESTATUS[0]}" -ne 0 ] && { echo "training failed"; exit 1; }

echo "=== [2/2] score on events [$TEST_SKIP,+$TEST_EV), pick a threshold, bake the header ==="
rt_run python3 "$RT_MODELS/train_triplet_dnn.py" finalize "${SPECS[@]}" \
  --bank "$BANK" --device "$RT_DEVICE" --force-variant "$VARIANT" \
  --skip-events "$TEST_SKIP" --max-events "$TEST_EV" --chunk-events 10 \
  --threshold-rule "$THRESHOLD_RULE" --survival-floor "$SURVIVAL_FLOOR" $RS $BT \
  --mem-guard-gb 260 --mem-log "$LOGDIR/retrain_ram.log" \
  --json-out "$LOGDIR/retrain_${VARIANT}.json" \
  $W $EXTRA 2>&1 | tee "$LOGDIR/retrain_finalize_${VARIANT}.log"
[ "${PIPESTATUS[0]}" -ne 0 ] && { echo "scoring/baking failed"; exit 1; }

if [ "$RT_DRYRUN" -eq 0 ]; then
  echo "=== baked header ==="
  grep -E "kNFeat|kDefaultThreshold|^// " "$HEADER" | head -12
fi

rt_report "Triplet gate retrained ($BANK iteration)"
rt_r_produced "$HEADER   (written in place)" \
              "scores, plots and logs: $OUTDIR"
rt_r_deploy "the weights are compiled into the kernel, so rebuild:" \
            "  scram b code-format && scram b -j" \
            "(never rebuild while a cmsRun job is running: the shared libraries are replaced in place)" \
            "code-format reformats the header a bake just wrote; its numbers do not change."
if [ "$CURRENT_SRC" = "the producer configuration" ]; then
  rt_r_update "$CA_CFI" \
              "  tripletDNNThreshold -- $CURRENT_THR today; re-scan it in situ after this retrain"
else
  rt_r_update "$CA_CFI" \
              "  tripletDNNThreshold is not set there, so the value baked into the header is what" \
              "  runs ($BAKE_THR). Set it explicitly if the in-situ scan lands somewhere else."
fi
