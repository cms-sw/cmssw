#!/bin/bash
# =============================================================================
# retrain_prompt.sh -- retrain the machine-learned models of the PROMPT pixel
# track iteration (hltPhase2PixelTracksSoAWithStubs).
# =============================================================================
# Three models, trained in this order. Each decides on the population the
# previous one leaves behind, so every dataset is produced with the upstream
# models already deployed and built; the steps refuse to run out of order where
# that is detectable and always print the rebuild gate. Retrain the prompt
# iteration BEFORE the displaced one: the displaced iteration runs on the hits
# the prompt one did not use (see retrain_displaced.sh), and the merged
# selector after both (see retrain_merged.sh).
#
#   step      model                        deployed as                     action
#   --------  ---------------------------  ------------------------------  --------------
#   triplet   per-triplet gate (in kernel) CATripletDNNWeights_prompt.h    rebuild
#   gate      track DNN, loose->tight      CATrackDNNWeights_prompt.h      rebuild
#   hp        final high-purity selector   data/.../prompt_tree31_<tag>_<date>.bin   file swap
#
# USAGE
#   ./retrain_prompt.sh <step> [options]
#
#   steps
#     dump      produce a training dataset            (--for triplet|gate|hp)
#     triplet   train + bake the per-triplet gate     [rebuild afterwards]
#     gate      train + bake the track DNN            [rebuild afterwards]
#     hp        train + stage the high-purity model   [file swap, no rebuild]
#               the 31-feature forest the chain runs, with the recipe of the deployed file;
#               --class both|forest|mlp runs the four-arm comparison of the alternatives
#               instead (a study, not the deploy path)
#     all       triplet, then the instructions to continue
#
#   options
#     --work DIR        where datasets, models and logs are written        (required)
#     --input LIST      event files to reconstruct when a dataset has to be produced:
#                       a comma-separated list, a directory, or a text file of paths.
#                       Per-sample inputs: --input name=files,name2=files2, or one
#                       --input per --sample name, paired in order.
#     --dataset FILE    use an already-produced dataset instead (repeatable / comma list)
#     --sample NAMES    dataset label(s), several in one quoted string; the files are
#                       trackNano_<sample>_<quality>.root / tripletNano_<sample>.root in
#                       --work, and a training step trains on their union  (default ttbarPU)
#     --events N        events per dataset; applies to the dump it is on
#     --threshold X     bake this decision threshold; 'rule' = the value the trainer
#                       derives                     (default: the value the chain runs today)
#     --bake-recall R   (gate) bake the derived point at this displacement-weighted recall,
#                       0.99 or 0.995; implies --threshold rule (see retrain_track_dnn.sh)
#     --label L         (gate) truth-label definition, mtv or legacy (see retrain_track_dnn.sh)
#     --class C         (hp) forest31 = the deployed family, one 31-feature forest (default);
#                       both|forest|mlp = the four-arm comparison (mlp17, mlp27, forest17,
#                       forest27, or one family of it), whose winner is staged for in-situ
#                       checks only
#     --threads N       reconstruction threads for the dumps                    (default 8)
#     --device DEV      torch device for training                         (default cuda:0)
#     --chassis MODULE  base reconstruction config to dump on top of; by default one is
#                       generated from --input with make_base_config.sh
#     --deploy          (hp) copy the staged model into the data directory
#     --dry-run         print the commands without running them
#
# EXAMPLES
#   ./retrain_prompt.sh all      --work /w --input /data/ttbar --events 1000
#   ./retrain_prompt.sh gate     --work /w --dataset /w/trackNano_ttbarPU_loose.root
#   ./retrain_prompt.sh hp       --work /w --input /data/ttbar
#   ./retrain_prompt.sh hp       --work /w --input /data/ttbar --class both   # the comparison
#   ./retrain_prompt.sh dump     --work /w --input /data/ttbar --for hp
#   # three samples from three inputs, one dump per stage (see README.md, "Inputs"):
#   ./retrain_prompt.sh dump --for gate --work /w --sample "ttbarPU displacedPU qcdPU" \
#       --input ttbarPU=/data/ttbar,displacedPU=/data/susy,qcdPU=/data/qcd --events 300
#
# ENVIRONMENT (tuning; the defaults are the ones in use)
#   hp step: WP_RULE (uniform | global | profile, see train_merged_forest.py --wp-rule),
#            RECALL (0.995), EFF_WEIGHT (4.0), LABEL (mtv), TAG (wp, the tag in the file
#            name), XGB_THREADS (16), MODEL_NAME (the date in the file name, today),
#            TRAIN_EXTRA (further train_merged_forest.py options, e.g. "--ntrees 200" for a
#            quick check).
#            Comparison only (--class both|forest|mlp): ARMS (explicit candidate list;
#            setting it selects the comparison), SELECT_RECALL (0.99), EPOCHS (250),
#            DEPTH (8), NTREES (800), THR_RECALL (0.99), MAX_ROWS (subsample the cache)
#   triplet step: TRAIN_EV, TEST_SKIP, TEST_EV, VARIANT, THRESHOLD_RULE, SURVIVAL_FLOOR,
#                 REF_SURVIVAL, EPOCHS, WORKERS  -- see retrain_triplet_gate.sh
#   gate step:    NAME  -- see retrain_track_dnn.sh
#   everywhere:   MEM_GUARD_GB (300)
#
# WORKING POINTS. The threshold a trainer prints is an offline starting point; the deployed
# values come from a scan of the full reconstruction (efficiency ceiling, combinatorial load,
# container occupancy, fake load to the next stage). By default each step bakes the value the
# chain runs today, so a retrain with no options changes the model and nothing else.
# =============================================================================
set -euo pipefail
# shellcheck source=retrain_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/retrain_common.sh"

rt_usage() { sed -n '/^# USAGE/,/^# ====/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//;$d'; }

rt_parse_args "$@"
STEP=$(rt_canonical_step "${RT_STEP:-}")
case "${STEP:-}" in
  dump|triplet|gate|hp|all) ;;
  "") rt_die "give a step: dump | triplet | gate | hp | all  (see --help)" ;;
  *)  rt_die "unknown step '$STEP' (dump | triplet | gate | hp | all)" ;;
esac

BANK=prompt
SAMPLES=${RT_SAMPLE:-${SAMPLES:-ttbarPU}}
CLASS=${RT_CLASS:-${MODEL_CLASS:-forest31}}
CA_CFI=$(rt_ca_cfi $BANK)
HP_CFI=$(rt_hp_cfi $BANK)
MODEL_TAG=${MODEL_NAME:-$(date +%Y%m%d)}

rt_require_work

# ---- dataset locations ------------------------------------------------------
track_nano() { local q=$1 s; for s in $SAMPLES; do echo "$RT_WORK/trackNano_${s}_${q}.root"; done; }
triplet_nano() { local s; for s in $SAMPLES; do echo "$RT_WORK/tripletNano_${s}.root"; done; }

# Datasets given with --dataset win; otherwise use the standard names in the
# working directory and produce the ones that are not there yet. The result is
# left in DATASETS.
DATASETS=()
resolve_datasets() {  # $1 = kind (triplet|track), $2 = quality (track only)
  local kind=$1 quality=${2:-} f s i
  if [ ${#RT_DATASETS[@]} -gt 0 ]; then DATASETS=("${RT_DATASETS[@]}"); return 0; fi
  if [ "$kind" = triplet ]; then mapfile -t DATASETS < <(triplet_nano)
  else mapfile -t DATASETS < <(track_nano "$quality"); fi
  local missing=0
  for f in "${DATASETS[@]}"; do [ -f "$f" ] || missing=1; done
  if [ "$missing" -eq 1 ]; then
    # A chassis carries its own input file list, so it is enough on its own.
    rt_can_produce ||
      rt_die "no dataset: pass --dataset <file>, or --input <event files> (or --chassis <module>, which carries its own) so one can be produced"
    if [ "$kind" = triplet ]; then rt_require_dump_build; fi
    rt_resolve_chassis
    i=0
    for s in $SAMPLES; do
      if [ ! -f "${DATASETS[$i]}" ]; then
        if [ "$kind" = triplet ]; then rt_dump_triplet_nano $BANK "$s" "${DATASETS[$i]}" "$i"
        else rt_dump_track_nano $BANK "$quality" "$s" "${DATASETS[$i]}" "$i"; fi
      fi
      i=$((i + 1))
    done
  fi
  return 0
}

# ---- steps ------------------------------------------------------------------
step_dump() {
  local what=${RT_FOR:-}
  case "$what" in
    triplet|gate|hp) ;;
    "") rt_die "say which dataset to produce: --for triplet | gate | hp" ;;
    *)  rt_die "--for must be triplet, gate or hp (got '$what'); the merged dataset is produced by retrain_merged.sh" ;;
  esac
  rt_require_no_cmsrun
  rt_cmsenv
  rt_resolve_chassis
  local s i
  case "$what" in
    triplet)
      rt_require_dump_build
      i=0; for s in $SAMPLES; do rt_dump_triplet_nano $BANK "$s" "$RT_WORK/tripletNano_${s}.root" "$i"; i=$((i + 1)); done
      rt_report "Triplet-gate dataset ready"
      rt_r_produced "$(triplet_nano | tr '\n' ' ')"
      rt_r_next "./retrain_prompt.sh triplet --work $RT_WORK"
      ;;
    gate)
      i=0; for s in $SAMPLES; do rt_dump_track_nano $BANK loose "$s" "$RT_WORK/trackNano_${s}_loose.root" "$i"; i=$((i + 1)); done
      rt_report "Track-DNN dataset ready (quality 'loose', track DNN switched off at dump time)"
      rt_r_produced "$(track_nano loose | tr '\n' ' ')"
      rt_r_next "./retrain_prompt.sh gate --work $RT_WORK"
      ;;
    hp)
      i=0; for s in $SAMPLES; do rt_dump_track_nano $BANK tight "$s" "$RT_WORK/trackNano_${s}_tight.root" "$i"; i=$((i + 1)); done
      rt_report "High-purity dataset ready (quality 'tight', full deployed chain)"
      rt_r_produced "$(track_nano tight | tr '\n' ' ')"
      rt_r_next "./retrain_prompt.sh hp --work $RT_WORK"
      ;;
  esac
}

step_triplet() {
  rt_require_no_cmsrun
  rt_cmsenv
  resolve_datasets triplet; local -a nanos=("${DATASETS[@]}")
  local -a args=(--bank $BANK --work "$RT_WORK" --device "$RT_DEVICE")
  [ -n "$RT_THRESHOLD" ] && args+=(--threshold "$RT_THRESHOLD")
  [ "$RT_DRYRUN" -eq 1 ] && args+=(--dry-run)
  local n; for n in "${nanos[@]}"; do args+=(--dataset "$n"); done
  "$RT_MODELS/retrain_triplet_gate.sh" "${args[@]}"
}

step_gate() {
  rt_require_no_cmsrun
  rt_cmsenv
  resolve_datasets track loose; local -a nanos=("${DATASETS[@]}")
  local -a args=(--bank $BANK --work "$RT_WORK" --device "$RT_DEVICE")
  [ -n "$RT_THRESHOLD" ] && args+=(--threshold "$RT_THRESHOLD")
  [ -n "$RT_BAKE_RECALL" ] && args+=(--bake-recall "$RT_BAKE_RECALL")
  [ -n "$RT_LABEL" ] && args+=(--label "$RT_LABEL")
  [ "$RT_DRYRUN" -eq 1 ] && args+=(--dry-run)
  local n; for n in "${nanos[@]}"; do args+=(--dataset "$n"); done
  "$RT_MODELS/retrain_track_dnn.sh" "${args[@]}"
}

# The selector the prompt chain runs is a 31-feature forest
# (PixelTrackForestHighPuritySelector, useHitFeatures=True, PixelTrackFeaturesSoA
# columns 1-31). By default the hp step trains exactly that, with the recipe of
# the deployed file; the four-arm comparison of the alternatives is a study
# behind --class both|forest|mlp.
step_hp() {
  rt_require_no_cmsrun
  rt_cmsenv
  resolve_datasets track tight; local -a nanos=("${DATASETS[@]}")
  local guard; guard=$(rt_start_guard "$RT_WORK/retrain_prompt_ram.log" "prompt hp")
  trap "kill $guard 2>/dev/null || true" EXIT
  if [ -n "${ARMS:-}" ] || [ "$CLASS" != forest31 ]; then hp_bakeoff "${nanos[@]}"
  else hp_deploy "${nanos[@]}"; fi
}

# The deploy path: the per-arm cache in the selector's column order, the trainer
# with the prompt recipe, the compact export, staged under the trainer's own name
# prompt_tree31_<tag>_<date>.bin.
hp_deploy() {  # $@ = tight datasets
  local cache="$RT_WORK/prompt31_cache.npz" outdir="$RT_WORK/prompt_forest"
  local label="${LABEL:-mtv}" tag="${TAG:-wp}"
  local deployed hpmod thr
  deployed=$(rt_deployed_model "$HP_CFI")
  # Read the working point off the producer the stub chain actually runs, not the
  # first one declared in the file (the Torch baseline sits above it).
  hpmod=$(rt_hp_module "$HP_CFI")
  thr=$(rt_cfi_double "$HP_CFI" scoreThreshold "$hpmod")

  echo ">>> [1/3] build the 31-feature cache -> $cache"
  # The 31 features as a prefix, in the order the selector reads them, with y_mtv
  # (matched to any TrackingParticle, the target) and y_eff (the recall axis).
  rt_run python3 "$RT_MODELS/nano_loader.py" cache-arm "$cache" "$@" --prefix TrkPrompt --label "$label"

  echo ">>> [2/3] train the forest -> $outdir"
  rt_wp_args prompt "$cache" "$RT_DATA/${deployed:-none}" "$thr" "$RT_WORK/prompt_reference_profile.json" \
    --label "$label"
  # The prompt recipe: the trainer's defaults (depth 12, up to 450 trees, two-pass tail
  # re-weight, early stopping on the working-point rule), true high-pT tracks up-weighted
  # (--eff-weight 4), full-precision values in the exported file (--no-fp16).
  local -a train=(--cache "$cache" --out "$outdir" --feats 31 --label "$label"
                  --recall "${RECALL:-0.995}" --arm prompt --tag "$tag" --date "$MODEL_TAG"
                  --threads "${XGB_THREADS:-16}" --eff-weight "${EFF_WEIGHT:-4.0}" --no-fp16
                  "${RT_WP_ARGS[@]}")
  # shellcheck disable=SC2206
  [ -n "${TRAIN_EXTRA:-}" ] && train+=(${TRAIN_EXTRA})
  rt_run python3 "$RT_TRAINER" "${train[@]}"

  echo ">>> [3/3] stage the compact model"
  local out; out=$(rt_forest_bin "$outdir" prompt 31 "$tag" "$MODEL_TAG")
  [ "$RT_DEPLOY" -eq 1 ] && rt_run cp "$out" "$RT_DATA/"

  rt_report "Prompt high-purity selector retrained"
  rt_r_produced "$out" \
                "training summary: $outdir/result.json   threshold map: $outdir/thrmap.txt" \
                "cache: $cache"
  rt_r_forest_deploy "$out" "$deployed"
  rt_r_update \
    "$HP_CFI" \
    "  model          = cms.FileInPath('RecoTracker/FinalTrackSelectors/data/" \
    "                     PixelTrackTorchHighPuritySelector/$(basename "$out")')" \
    "  useHitFeatures = cms.bool(True)   (unchanged: a 31-feature model reads the hit features)" \
    "  scoreThreshold = cms.double(<re-scan in situ>)   currently ${thr:-?}${hpmod:+  (module $hpmod)}" \
    "                   the trainer's starting point: $(rt_forest_wp "$outdir")" \
    "  (the prompt arm keeps scoreThresholdLowDxy = -1.0: no displacement ramp)"
  echo " The threshold the trainer printed is an offline starting point. Re-scan it by"
  echo " running the reconstruction and the track validation before pinning it."
}

# The comparison: the four candidates of prompt_hp_bakeoff.py (17 or 27 features,
# forest or neural network) trained on one cache and one event split and ranked at
# one working-point rule. None of them is the 31-feature model the chain runs, so
# the winner is staged for in-situ checks, and deploying it is a change of model.
hp_bakeoff() {  # $@ = tight datasets
  local arms
  case "$CLASS" in
    both|forest31) arms=${ARMS:-mlp17,mlp27,forest17,forest27} ;;
    forest)        arms=${ARMS:-forest17,forest27} ;;
    mlp)           arms=${ARMS:-mlp17,mlp27} ;;
    *) rt_die "--class must be forest31, both, forest or mlp (got '$CLASS')" ;;
  esac

  local cache="$RT_WORK/prompt27_cache.npz"

  echo ">>> [1/3] build the shared feature cache -> $cache"
  # One cache for all candidates: identical rows and an identical event split for
  # every candidate, which is what makes the comparison meaningful.
  rt_run python3 "$RT_MODELS/nano_loader.py" cache-prompt27 "$cache" "$@"

  echo ">>> [2/3] train and compare the candidate models: $arms"
  # Both model families are trained on the same cache and the same split, so a
  # change of family is a measured decision rather than a habit.
  local -a bo=(--cache "$cache" --outdir "$RT_WORK" --arms "$arms"
               --select-recall "${SELECT_RECALL:-0.99}" --device "$RT_DEVICE"
               --epochs "${EPOCHS:-250}" --depth "${DEPTH:-8}" --ntrees "${NTREES:-800}"
               --threads "${XGB_THREADS:-16}")
  [ -n "${MAX_ROWS:-}" ] && bo+=(--max-rows "$MAX_ROWS")
  rt_run python3 "$RT_MODELS/prompt_hp_bakeoff.py" "${bo[@]}"
  [ "$RT_DRYRUN" -eq 1 ] && return 0

  local wclass warm wart wfeat
  wclass=$(python3 -c "import json;print(json.load(open('$RT_WORK/WINNER.json'))['class'])")
  warm=$(python3 -c "import json;print(json.load(open('$RT_WORK/WINNER.json'))['winner'])")
  wart=$(python3 -c "import json;print(json.load(open('$RT_WORK/WINNER.json'))['artifact'])")
  wfeat=$(python3 -c "import json;print(json.load(open('$RT_WORK/WINNER.json'))['n_feats'])")

  echo ">>> [3/3] stage the best candidate: $warm ($wclass, $wfeat features)"
  # Read the working point off the producer the stub chain actually runs, not the
  # first one declared in the file (the Torch baseline sits above it).
  local hpmod out; hpmod=$(rt_hp_module "$HP_CFI")
  if [ "$wclass" = forest ]; then
    # Named by its own width, not after the deployed file: a 17- or 27-feature forest
    # is not the 31-feature family the configuration loads. Full-precision values,
    # the convention of the deployed prompt file.
    out="$RT_WORK/prompt_tree${wfeat}_bakeoff_${MODEL_TAG}.bin"
    rt_run python3 "$RT_MODELS/export_compact_tree.py" --booster "$RT_WORK/$wart" \
      --out-bin "$out" --no-fp16
    [ "$RT_DEPLOY" -eq 1 ] && rt_run cp "$out" "$RT_DATA/"
    rt_report "Prompt high-purity comparison done: best candidate $warm (forest, $wfeat features)"
    rt_r_produced "$out" "comparison table: $RT_WORK/HP_BAKEOFF.txt" "winner: $RT_WORK/WINNER.json"
    rt_r_deploy "cp $out $RT_DATA/" "no rebuild needed -- the selector reads the file at run time" \
                "(a $wfeat-feature model is not the 31-feature family the chain runs: deploying" \
                " it is a change of model, to be validated as one, not a retrain)"
    rt_r_update \
      "$HP_CFI" \
      "  model          = cms.FileInPath('RecoTracker/FinalTrackSelectors/data/" \
      "                     PixelTrackTorchHighPuritySelector/$(basename "$out")')" \
      "  useHitFeatures = cms.bool(<True for a 27-feature model; a 17-feature one does not need it>)" \
      "  scoreThreshold = cms.double(<re-scan in situ>)   currently $(rt_cfi_double "$HP_CFI" scoreThreshold "$hpmod")${hpmod:+  (module $hpmod)}" \
      "  (the prompt arm keeps scoreThresholdLowDxy = -1.0: no displacement ramp)"
  else
    out="$RT_WORK/pixel_track_classifier_${MODEL_TAG}_${warm}.pt"
    # Exported from the comparison's own model on the same cache and the same
    # split, so the printed threshold belongs to the numbers in the table.
    rt_run python3 "$RT_MODELS/train_prompt_hp_nano.py" finalize --cache "$cache" \
      --prefix TrkPrompt --name "$warm" --ts-name "$(basename "$out")" \
      --thr-recall "${THR_RECALL:-0.99}"
    [ "$RT_DEPLOY" -eq 1 ] && rt_run cp "$out" "$RT_DATA/"
    rt_report "Prompt high-purity comparison done: best candidate $warm (neural network, $wfeat features)"
    rt_r_produced "$out" "16-bit default, plus a 32-bit sibling (_fp32.pt)" \
                  "comparison table: $RT_WORK/HP_BAKEOFF.txt"
    rt_r_deploy "cp $out $RT_DATA/" "no rebuild needed" \
                "(a neural network is not the forest family the chain runs: deploying it is a" \
                " change of model, to be validated as one, not a retrain)"
    rt_r_update "$HP_CFI" \
      "  switch the producer back to 'PixelTrackTorchHighPuritySelector@alpaka'," \
      "  point model = cms.FileInPath(...) at $(basename "$out")," \
      "  and set scoreThreshold after an in-situ scan."
  fi
  echo " The threshold in the table is an offline starting point. Re-scan it by"
  echo " running the reconstruction and the track validation before pinning it."
}

step_all() {
  step_triplet
  echo
  echo "============================================================================"
  echo " The remaining steps need the new triplet gate compiled in and the datasets"
  echo " re-produced underneath it. Continue with:"
  echo "   scram b code-format && scram b -j"
  echo "   ./retrain_prompt.sh gate --work $RT_WORK --input <event files>"
  echo "   scram b code-format && scram b -j"
  echo "   ./retrain_prompt.sh hp   --work $RT_WORK --input <event files>"
  echo "============================================================================"
}

case "$STEP" in
  dump)    step_dump ;;
  triplet) step_triplet ;;
  gate)    step_gate ;;
  hp)      step_hp ;;
  all)     step_all ;;
  *) rt_die "unknown step '$STEP' (dump | triplet | gate | hp | all)" ;;
esac
