#!/usr/bin/env bash
set -euo pipefail
 
: <<'COMMENT'
Usage:
  run_tau_decaymode_plots.sh [HLT|RECO] [SUBDIR] [DQM_FILE] [MATCHING_DR]
 
Arguments:
  [HLT|RECO]   Which DQM path to use. Default: HLT
  [SUBDIR]     Optional TauValidation subdirectory, e.g. CutWP_VSjet0 or CutID_VSjet0p70
               Leave empty to use TauValidation/DecayModes.
               With SUBDIR, read TauValidation/SUBDIR/DecayModes.
  [DQM_FILE]   Optional harvested DQM ROOT file.
               Default: DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root
  [MATCHING_DR] Matching radius used to PRODUCE the input, for labels only. Default: 0.3
 
Examples:
  ./run_tau_decaymode_plots.sh HLT
  ./run_tau_decaymode_plots.sh RECO
  ./run_tau_decaymode_plots.sh HLT CutWP_VSjet0
  ./run_tau_decaymode_plots.sh HLT CutID_VSjet0p70
  ./run_tau_decaymode_plots.sh HLT "" step3_hlt_HARVESTING_ALL.root
  ./run_tau_decaymode_plots.sh RECO CutWP_VSjet0 /path/to/step3_reco_HARVESTING.root
 
This script overlays tau validation quantities split by decay mode.
 
The default HLT and RECO configuration uses DeltaR = 0.3.
COMMENT

set -euo pipefail
if [[ -z "${CMSSW_BASE:-}" ]]; then
    echo "ERROR: run this script from an initialized CMSSW area (cmsenv)." >&2
    exit 2
fi
if [[ "$#" -gt 4 ]]; then
    echo "Usage: $0 [HLT|RECO] [SUBDIR] [DQM_FILE] [MATCHING_DR]" >&2
    exit 2
fi
STEP_IN="${1:-HLT}"
SUB_DIR="${2:-}"
DQM_FILE_HLT="/eos/user/s/smeriano/UCLouvain/Authorship_task/CMSSW_20_1_0_pre3/src/local_run_28_Sept_v1_condor_HLT_condor_HLT_HARVESTING_ALL_DELTAR_0p3/step3_hlt_HARVESTING_local.root"
DQM_FILE_RECO="/eos/user/s/smeriano/UCLouvain/Authorship_task/CMSSW_20_1_0_pre3/src/local_run_28_Sept_v1_condor_RECO_HARVESTING_ALL_D128_DELTAR_0p3/step4_reco_HARVESTING_local.root"
DQM_FILE="${3:-}"
MATCHING_DR="${4:-0.3}"
STEP_UPPER="${STEP_IN^^}"
case "$STEP_UPPER" in
    HLT)
        STEP="HLT"
        BASE_DIR="DQMData/Run 1/HLT/Run summary/Tau/TauValidation"
        DQM_FILE="${DQM_FILE:-$DQM_FILE_HLT}"
        ;;
    RECO)
        STEP="Reco"
        BASE_DIR="DQMData/Run 1/Tau/Run summary/TauValidation"
        DQM_FILE="${DQM_FILE:-$DQM_FILE_RECO}"
        ;;
    *)
        echo "Invalid step: $STEP_IN"
        echo "Use HLT or RECO"
        exit 1
        ;;
esac
SCRIPT_DIR="${CMSSW_BASE}/src/Validation/RecoTau/scripts"
MAKE_COMPARISON="${SCRIPT_DIR}/makeComparisonPlots.py"
MAKE_TAU_VALIDATION="${SCRIPT_DIR}/makeTauValidationPlots.py"
ENERGY_TEXT="Ten Tau | 14 TeV"
LABEL_TEXT="${STEP} Tau validation, $\Delta R < ${MATCHING_DR}$"
if [ -n "$SUB_DIR" ]; then
    SELECTED_DIR="${BASE_DIR}/${SUB_DIR}/DecayModes"
    OUTDIR="TauValidationPlots/DecayModes_${STEP_UPPER}_${SUB_DIR}"
    LABEL_TEXT="${LABEL_TEXT}, ${SUB_DIR}"
else
    SELECTED_DIR="${BASE_DIR}/DecayModes"
    OUTDIR="TauValidationPlots/DecayModes_${STEP_UPPER}_NoCut"
fi
OUTDIR_SUMMARY="${OUTDIR}/Summary"
PT_REBIN="0,5,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,220,240,260,280,300,340,380,400"
ETA_REBIN="-2.4,-2.0,-1.6,-1.2,-0.8,-0.4,0.0,0.4,0.8,1.2,1.6,2.0,2.4"
PHI_REBIN="-3.5,-2.8,-2.1,-1.4,-0.7,0.0,0.7,1.4,2.1,2.8,3.5"
GEN_DECAY_MODES=(
    "oneProng0Pi0"
    "oneProng1Pi0"
    "oneProngOther"
    "threeProng0Pi0"
    "threeProng1Pi0"
    "threeProngOther"
)
GEN_DECAY_LABELS=(
    '1-prong $0\pi^{0}$'
    '1-prong $1\pi^{0}$'
    '1-prong other'
    '3-prong $0\pi^{0}$'
    '3-prong $1\pi^{0}$'
    '3-prong other'
)
RECO_DECAY_MODES=(
    "oneProng0Pi0"
    "oneProng1Pi0"
    "oneProngOther"
    "threeProng0Pi0"
    "threeProng1Pi0"
    "threeProngOther"
)
RECO_DECAY_LABELS=(
    '1-prong $0\pi^{0}$'
    '1-prong $1\pi^{0}$'
    '1-prong other'
    '3-prong $0\pi^{0}$'
    '3-prong $1\pi^{0}$'
    '3-prong other'
)
if [ "${#GEN_DECAY_MODES[@]}" -ne "${#GEN_DECAY_LABELS[@]}" ] || \
   [ "${#RECO_DECAY_MODES[@]}" -ne "${#RECO_DECAY_LABELS[@]}" ]; then
    echo "ERROR: decay mode and label arrays have different lengths." >&2
    exit 1
fi
has_hist() {
    local hist="$1"
    rootls "$DQM_FILE:$hist" >/dev/null 2>&1
}
join_by_comma() {
    local IFS=","
    echo "$*"
}
run_cmd() {
    echo
    echo "Running:"
    echo "$*"
    "$@"
}
make_decaymode_overlay() {
    local metric="$1"
    local variable="$2"
    local name="$3"
    local xlabel="$4"
    local ylabel="$5"
    local xlim="$6"
    local ylim="$7"
    local rebin="$8"
    local mode_type="$9"
    local inverted="${10:-0}"
    local files=()
    local hists=()
    local labels=()
    local modes=()
    local mode_labels=()
    if [ "$mode_type" = "gen" ]; then
        modes=("${GEN_DECAY_MODES[@]}")
        mode_labels=("${GEN_DECAY_LABELS[@]}")
    else
        modes=("${RECO_DECAY_MODES[@]}")
        mode_labels=("${RECO_DECAY_LABELS[@]}")
    fi
    for i in "${!modes[@]}"; do
        local dm="${modes[$i]}"
        local hist="${SELECTED_DIR}/${metric}_${dm}_vs_${variable}"
        if has_hist "$hist"; then
            files+=("$DQM_FILE")
            hists+=("$hist")
            labels+=("${mode_labels[$i]}")
        else
            echo "Skipping missing histogram: $hist"
        fi
    done
    if [ "${#hists[@]}" -eq 0 ]; then
        echo "No valid histograms for ${metric}_*_vs_${variable}. Skipping."
        return
    fi
    local cmd=(
        python3 "$MAKE_COMPARISON"
        --files "$(join_by_comma "${files[@]}")"
        --hists "$(join_by_comma "${hists[@]}")"
        --labels "$(join_by_comma "${labels[@]}")"
        --xlabel "$xlabel"
        --ylabel "$ylabel"
        --leg-title "$LABEL_TEXT"
        --energy-text "$ENERGY_TEXT"
        --odir "$OUTDIR"
        --name "$name"
    )
    if [ -n "$xlim" ]; then
        cmd+=("--xlim=${xlim}")
    fi
    if [ -n "$ylim" ]; then
        cmd+=("--ylim=${ylim}")
    fi
    if [ -n "$rebin" ]; then
        cmd+=("--rebin=${rebin}")
    fi
    if [ "$inverted" -eq 1 ]; then
        cmd+=(--inverted)
    fi
    run_cmd "${cmd[@]}"
}
make_fake_source_decaymode_plots() {
    local variable="$1"
    local xlabel="$2"
    local xlim="$3"
    local rebin="$4"
    local sources=("Electron" "Muon" "Jet" "Other")
    local source_labels=("Electron fakes" "Muon fakes" "Jet fakes" "Other fakes")
    for i in "${!RECO_DECAY_MODES[@]}"; do
        local dm="${RECO_DECAY_MODES[$i]}"
        local dm_label="${RECO_DECAY_LABELS[$i]}"
        local files=()
        local hists=()
        local labels=()
        for j in "${!sources[@]}"; do
            local hist="${SELECTED_DIR}/Fake${sources[$j]}_${dm}_vs_${variable}"
            if has_hist "$hist"; then
                files+=("$DQM_FILE")
                hists+=("$hist")
                labels+=("${source_labels[$j]}")
            fi
        done
        if [ "${#hists[@]}" -eq 0 ]; then
            echo "No fake-source profiles found for ${dm} vs ${variable}"
            continue
        fi
        local cmd=(
            python3 "$MAKE_COMPARISON"
            --files "$(join_by_comma "${files[@]}")"
            --hists "$(join_by_comma "${hists[@]}")"
            --labels "$(join_by_comma "${labels[@]}")"
            --xlabel "$xlabel"
            --ylabel "Fake rate"
            --ylim=0,1.2
            --leg-title "${LABEL_TEXT}, ${dm_label}"
            --energy-text "$ENERGY_TEXT"
            --odir "$OUTDIR"
            --name "FakeSources_${dm}_vs_${variable}"
        )
        if [ -n "$xlim" ]; then
            cmd+=("--xlim=${xlim}")
        fi
        if [ -n "$rebin" ]; then
            cmd+=("--rebin=${rebin}")
        fi
        run_cmd "${cmd[@]}"
    done
}
make_response_decaymode_overlay() {
    local response_base="$1"
    local variable="$2"
    local name="$3"
    local xlabel="$4"
    local ylabel="$5"
    local xlim="$6"
    local ylim="$7"
    local rebin="$8"
    local files=()
    local hists=()
    local labels=()
    for i in "${!GEN_DECAY_MODES[@]}"; do
        local dm="${GEN_DECAY_MODES[$i]}"
        local hist="${SELECTED_DIR}/${response_base}_${dm}_RecoOverGen_vs_${variable}_Mean"
        if has_hist "$hist"; then
            files+=("$DQM_FILE")
            hists+=("$hist")
            labels+=("${GEN_DECAY_LABELS[$i]}")
        else
            echo "Skipping missing histogram: $hist"
        fi
    done
    if [ "${#hists[@]}" -eq 0 ]; then
        echo "No valid response histograms for ${response_base}_*_${variable}. Skipping."
        return
    fi
    local cmd=(
        python3 "$MAKE_COMPARISON"
        --files "$(join_by_comma "${files[@]}")"
        --hists "$(join_by_comma "${hists[@]}")"
        --labels "$(join_by_comma "${labels[@]}")"
        --xlabel "$xlabel"
        --ylabel "$ylabel"
        --leg-title "$LABEL_TEXT"
        --energy-text "$ENERGY_TEXT"
        --odir "$OUTDIR"
        --name "$name"
    )
    if [ -n "$xlim" ]; then
        cmd+=("--xlim=${xlim}")
    fi
    if [ -n "$ylim" ]; then
        cmd+=("--ylim=${ylim}")
    fi
    if [ -n "$rebin" ]; then
        cmd+=("--rebin=${rebin}")
    fi
    run_cmd "${cmd[@]}"
}
make_decaymode_summary_plot() {
    local metric="$1"
    local variable="$2"
    local name_prefix="$3"
    local xlabel="$4"
    local ylabel="$5"
    local xlim="$6"
    local ylim="$7"
    local rebin="$8"
    local mode_type="$9"
    local den_base="${10}"
    local num_base="${11}"
    local rate_label="${12}"
    local den_label="${13}"
    local num_label="${14}"
    local inverted="${15:-0}"
    local modes=()
    local mode_labels=()
    if [ "$mode_type" = "gen" ]; then
        modes=("${GEN_DECAY_MODES[@]}")
        mode_labels=("${GEN_DECAY_LABELS[@]}")
    else
        modes=("${RECO_DECAY_MODES[@]}")
        mode_labels=("${RECO_DECAY_LABELS[@]}")
    fi
    for i in "${!modes[@]}"; do
        local dm="${modes[$i]}"
        local dm_label="${mode_labels[$i]}"
        local den="${SELECTED_DIR}/${den_base}_${dm}_${variable}"
        local num="${SELECTED_DIR}/${num_base}_${dm}_${variable}"
        local rate="${SELECTED_DIR}/${metric}_${dm}_vs_${variable}"
        if ! has_hist "$den"; then
            echo "Skipping missing denominator: $den"
            continue
        fi
        if ! has_hist "$num"; then
            echo "Skipping missing numerator: $num"
            continue
        fi
        if ! has_hist "$rate"; then
            echo "Skipping missing rate: $rate"
            continue
        fi
        local cmd=(
            python3 "$MAKE_TAU_VALIDATION"
            --mode summary
            --files "$DQM_FILE"
            --den-hists "$den"
            --num-hists "$num"
            --rate-hists "$rate"
            --labels "$rate_label"
            --den-label "$den_label"
            --num-label "$num_label"
            --xlabel "$xlabel"
            --ylabel "$ylabel"
            --ylim="$ylim"
            --energy-text "$ENERGY_TEXT"
            --leg-title "${LABEL_TEXT}, ${dm_label}"
            --odir "$OUTDIR_SUMMARY"
            --name "${name_prefix}_${dm}"
        )
        if [ -n "$xlim" ]; then
            cmd+=("--xlim=${xlim}")
        fi
        if [ -n "$rebin" ]; then
            cmd+=("--rebin=${rebin}")
        fi
        if [ "$inverted" -eq 1 ]; then
            cmd+=(--inverted)
        fi
        run_cmd "${cmd[@]}"
    done
}
make_all_decaymode_summary_plots() {
    echo
    echo "Making per-decay-mode summary plots"
    make_decaymode_summary_plot "Eff" "pt"   "Summary_Eff_vs_pt"   'GenVis $\tau$ $p_{\mathrm{T}}$ [GeV]' "Efficiency" "0,400" "0,1.2" "$PT_REBIN"  "gen"  "genTau"  "genTauMatched"  "Efficiency" "Gen $\tau$'s"  "Gen $\tau$'s matched to reco $\tau$'s"
    make_decaymode_summary_plot "Eff" "eta"  "Summary_Eff_vs_eta"  'GenVis $\tau$ $\eta$'      "Efficiency" "-2.5,2.5" "0,1.2" "$ETA_REBIN" "gen"  "genTau"  "genTauMatched"  "Efficiency" "Gen $\tau$'s"  "Gen $\tau$'s matched to reco $\tau$'s"
    make_decaymode_summary_plot "Eff" "phi"  "Summary_Eff_vs_phi"  'GenVis $\tau$ $\phi$'      "Efficiency" "" "0,1.2" "$PHI_REBIN" "gen"  "genTau"  "genTauMatched"  "Efficiency" "Gen $\tau$'s"  "Gen $\tau$'s matched to reco $\tau$'s"
    make_decaymode_summary_plot "Eff" "mass" "Summary_Eff_vs_mass" 'GenVis $\tau$ mass [GeV]'  "Efficiency" "0,2" "0,1.2" "2"        "gen"  "genTau"  "genTauMatched"  "Efficiency" "Gen $\tau$'s"  "Gen $\tau$'s matched to reco $\tau$'s"
    make_decaymode_summary_plot "Split" "pt"   "Summary_Split_vs_pt"   'GenVis $\tau$ $p_{\mathrm{T}}$ [GeV]' "Split rate" "0,400" "0,1.2" "$PT_REBIN"  "gen" "genTau" "genTauMultiMatched" "Split rate" "Gen $\tau$'s" "Gen $\tau$'s matched to multiple reco $\tau$'s"
    make_decaymode_summary_plot "Split" "eta"  "Summary_Split_vs_eta"  'GenVis $\tau$ $\eta$'      "Split rate" "-2.5,2.5" "0,1.2" "$ETA_REBIN" "gen" "genTau" "genTauMultiMatched" "Split rate" "Gen $\tau$'s" "Gen $\tau$'s matched to multiple reco $\tau$'s"
    make_decaymode_summary_plot "Split" "phi"  "Summary_Split_vs_phi"  'GenVis $\tau$ $\phi$'      "Split rate" "" "0,1.2" "$PHI_REBIN" "gen" "genTau" "genTauMultiMatched" "Split rate" "Gen $\tau$'s" "Gen $\tau$'s matched to multiple reco $\tau$'s"
    make_decaymode_summary_plot "Split" "mass" "Summary_Split_vs_mass" 'GenVis $\tau$ mass [GeV]'  "Split rate" "0,2" "0,1.2" "2"        "gen" "genTau" "genTauMultiMatched" "Split rate" "Gen $\tau$'s" "Gen $\tau$'s matched to multiple reco $\tau$'s"
    make_decaymode_summary_plot "Dup" "pt"   "Summary_Dup_vs_pt"   '$\tau$ $p_{\mathrm{T}}$ [GeV]' "Duplicate rate" "0,400" "0,1.2" "$PT_REBIN"  "reco" "recoTau" "recoTauMultiMatched" "Duplicate rate" "Reco $\tau$'s" "Reco $\tau$'s matched to multiple gen $\tau$'s"
    make_decaymode_summary_plot "Dup" "eta"  "Summary_Dup_vs_eta"  '$\tau$ $\eta$'      "Duplicate rate" "-2.5,2.5" "0,1.2" "$ETA_REBIN" "reco" "recoTau" "recoTauMultiMatched" "Duplicate rate" "Reco $\tau$'s" "Reco $\tau$'s matched to multiple gen $\tau$'s"
    make_decaymode_summary_plot "Dup" "phi"  "Summary_Dup_vs_phi"  '$\tau$ $\phi$'      "Duplicate rate" "" "0,1.2" "$PHI_REBIN" "reco" "recoTau" "recoTauMultiMatched" "Duplicate rate" "Reco $\tau$'s" "Reco $\tau$'s matched to multiple gen $\tau$'s"
    make_decaymode_summary_plot "Dup" "mass" "Summary_Dup_vs_mass" '$\tau$ mass [GeV]'  "Duplicate rate" "0,2" "0,1.2" "2"        "reco" "recoTau" "recoTauMultiMatched" "Duplicate rate" "Reco $\tau$'s" "Reco $\tau$'s matched to multiple gen $\tau$'s"
}
if ! rootls "$DQM_FILE:$SELECTED_DIR" >/dev/null 2>&1; then
    echo "Cannot read decay-mode directory: $SELECTED_DIR" >&2
    echo "Check the input file and ROOT environment. Regenerate and harvest files using the DecayModes layout." >&2
    exit 1
fi
mkdir -p "$OUTDIR" "$OUTDIR_SUMMARY"
echo "Input file:"
echo "$DQM_FILE"
echo
echo "Selected DQM directory:"
echo "$SELECTED_DIR"
echo
echo "Making decay-mode plots for ${STEP_UPPER}, DeltaR < ${MATCHING_DR}"
make_all_decaymode_summary_plots
for variable in pt eta phi mass; do
    case "$variable" in
        pt)
            xlabel='GenVis $\tau$ $p_{\mathrm{T}}$ [GeV]'
            reco_xlabel='$\tau$ $p_{\mathrm{T}}$ [GeV]'
            xlim="0,400"
            rebin="$PT_REBIN"
            ;;
        eta)
            xlabel='GenVis $\tau$ $\eta$'
            reco_xlabel='$\tau$ $\eta$'
            xlim="-2.5,2.5"
            rebin="$ETA_REBIN"
            ;;
        phi)
            xlabel='GenVis $\tau$ $\phi$'
            reco_xlabel='$\tau$ $\phi$'
            xlim=""
            rebin="$PHI_REBIN"
            ;;
        mass)
            xlabel='GenVis $\tau$ mass [GeV]'
            reco_xlabel='$\tau$ mass [GeV]'
            xlim="0,2"
            rebin="2"
            ;;
    esac
    make_decaymode_overlay "Eff" "$variable" "Eff_vs_${variable}_byDecayMode" "$xlabel" "Efficiency" "$xlim" "0,1.2" "$rebin" "gen"
    make_decaymode_overlay "Split" "$variable" "Split_vs_${variable}_byDecayMode" "$xlabel" "Split rate" "$xlim" "0,1.2" "$rebin" "gen"
    make_decaymode_overlay "Dup" "$variable" "Dup_vs_${variable}_byDecayMode" "$reco_xlabel" "Duplicate rate" "$xlim" "0,1.2" "$rebin" "reco"
    make_response_decaymode_overlay "ResponsePt" "$variable" "ResponsePt_vs_${variable}_byDecayMode" "$xlabel" 'Mean $p_{T}^{reco}/p_{T}^{gen}$' "$xlim" "0,2" "$rebin"
    make_response_decaymode_overlay "ResponseMass" "$variable" "ResponseMass_vs_${variable}_byDecayMode" "$xlabel" 'Mean $m^{reco}/m^{gen}$' "$xlim" "0,2" "$rebin"
done
make_fake_source_decaymode_plots "pt"   '$\tau$ $p_{\mathrm{T}}$ [GeV]' "0,400" "$PT_REBIN"
make_fake_source_decaymode_plots "eta"  '$\tau$ $\eta$'                "-2.5,2.5" "$ETA_REBIN"
make_fake_source_decaymode_plots "phi"  '$\tau$ $\phi$'                 "" "$PHI_REBIN"
make_fake_source_decaymode_plots "mass" '$\tau$ mass [GeV]'             "0,2" "2"
echo
echo "Done."
