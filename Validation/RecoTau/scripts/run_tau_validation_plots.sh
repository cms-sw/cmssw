#!/usr/bin/env bash

: <<'COMMENT'

Usage: run_tau_validation_plots.sh [HLT|RECO]
  HLT   - use HLT DQM path and labels
  RECO  - use RECO DQM path and labels
  (no argument) - run both HLT and RECO

It assumes input DQM file 
Command example for HLT validation to obtain the DQM file (based on CMSSW_16_1_0_pre4):

cmsDriver.py step2 -s L1P2GT,HLT:75e33,VALIDATION:@hltValidation -n -1 --nThreads 0 \
 --conditions auto:phase2_realistic_T35 --datatier GEN-SIM-DIGI-RAW,DQMIO \
 --customise SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000 --eventcontent FEVTDEBUGHLT,DQMIO \
 --geometry ExtendedRun4D110 --era Phase2C17I13M9 --hltProcess HLTX --processName HLTX \
 --filein file:/eos/cms/store/relval/CMSSW_16_1_0_pre2/RelValTenTau_15_500/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_STD_Run4D110_PU-v1/2590000/01cbb197-f9d0-48a5-a634-c985f3b66373.root --fileout file:step2.root \
 --inputCommands="keep *, drop *_hlt*_*_HLT, drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT"

cmsDriver.py step3 -s HARVESTING:@hltValidation -n -1 \
 --conditions auto:phase2_realistic_T35 --mc --geometry ExtendedRun4D110 --era Phase2C17I13M9 \
 --filetype DQM --scenario pp --hltProcess HLTX --filein file:step2_inDQM.root --fileout file:step3.root
COMMENT

set -u

# Determine which steps to run
if [ "$#" -gt 0 ]; then
    STEPS=("$1")
else
    STEPS=("HLT" "RECO")
fi

# Common configuration
DQM_FILE="DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root"
SCRIPT_DIR="${CMSSW_BASE}/src/Validation/RecoTau/scripts"
MAKE_TAU_VALIDATION="${SCRIPT_DIR}/makeTauValidationPlots.py"
MAKE_COMPARISON="${SCRIPT_DIR}/makeComparisonPlots.py"
ENERGY_TEXT="Ten Tau (150 PU) | 14 TeV"

# Configuration for selected DeltaR and ID cut
SUB_DIR="CutWP_VSjet0" # "CutID_VSjet0p50"
OUTDIR_SUFFIX="CutWP_VSjet0" # "CutID_VSjet0p50"
LABEL_TEXT="Tau validation WP vs jet > 0"
OUTDIR_COMPARISON="TauValidationPlots/Summary_${OUTDIR_SUFFIX}_HLTvsRECO"

PT_REBIN="0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,220,240,260,280,300,320,340,360,380,400"
ETA_REBIN="-2.4,-2.0,-1.6,-1.2,-0.8,-0.4,0.0,0.4,0.8,1.2,1.6,2.0,2.4"
PHI_REBIN="-3.5,-2.8,-2.1,-1.4,-0.7,0.0,0.7,1.4,2.1,2.8,3.5"

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

make_summary_plot() {
    local variable="$1"
    local den_hist="$2"
    local num_hist="$3"
    local rate_hist="$4"
    local name="$5"
    local xlabel="$6"
    local ylabel="$7"
    local xlim="${8}"
    local ylim="${9}"
    local rebin="${10}"
    local rate_label="${11}"
    local den_label="${12}"
    local num_label="${13}"
    local extra_args="${14:-}"

    local den="${SELECTED_DIR}/${den_hist}"
    local num="${SELECTED_DIR}/${num_hist}"
    local rate="${SELECTED_DIR}/${rate_hist}"

    if ! has_hist "$den"; then
        echo "Skipping missing denominator: $den"
        return
    fi

    if ! has_hist "$num"; then
        echo "Skipping missing numerator: $num"
        return
    fi

    if ! has_hist "$rate"; then
        echo "Skipping missing rate: $rate"
        return
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
        --ylim "$ylim"
        --energy-text "$ENERGY_TEXT"
        --leg-title "${STEP} ${LABEL_TEXT}"
        --odir "${OUTDIR_SUMMARY}"
        --name "$name"
    )

    if [ -n "$xlim" ]; then
        cmd+=("--xlim=${xlim}")
    fi

    if [ -n "$rebin" ]; then
        cmd+=("--rebin=${rebin}")
    fi

    if [ -n "$extra_args" ]; then
        cmd+=("$extra_args")
    fi

    run_cmd "${cmd[@]}"
}

# Loop through each step
for step in "${STEPS[@]}"; do
    step_upper="${step^^}"
    case "$step_upper" in
        HLT)
            STEP="HLT"
            BASE_DIR="DQMData/Run 1/HLT/Run summary/Tau/TauValidation"
            ;;
        RECO)
            STEP="Reco"
            BASE_DIR="DQMData/Run 1/Tau/Run summary/TauValidation"
            ;;
        *)
            echo "Invalid step: $step"
            continue
            ;;
    esac

    # Setup paths for this step
    OUTDIR_SUMMARY="TauValidationPlots/Summary_${OUTDIR_SUFFIX}_${step_upper}"
    SELECTED_DIR="${BASE_DIR}/${SUB_DIR}"

    mkdir -p "$OUTDIR_SUMMARY"

    echo "Input file:"
    echo "$DQM_FILE"

    echo
    echo "Making summary plots for ${step_upper}"

    make_summary_plot "pt" "genTau_pt" "genTauMatched_pt" "Eff_vs_pt" "Efficiency_pt" 'Simulated $\tau$ $p_T$ [GeV]' "${STEP} Tau Efficiency" "0,400" "0,1.2" "2" "Efficiency" "Gen $\tau$'s" "Gen $\tau$'s matched to reco $\tau$'s"
    make_summary_plot "eta" "genTau_eta" "genTauMatched_eta" "Eff_vs_eta" "Efficiency_eta" 'Simulated $\tau$ $\eta$' "${STEP} Tau Efficiency" "-2.5,2.5" "0,1.2" "2" "Efficiency" "Gen $\tau$'s" "Gen $\tau$'s matched to reco $\tau$'s"
    make_summary_plot "phi" "genTau_phi" "genTauMatched_phi" "Eff_vs_phi" "Efficiency_phi" 'Simulated $\tau$ $\phi$' "${STEP} Tau Efficiency" "" "0,1.2" "2" "Efficiency" "Gen $\tau$'s" "Gen $\tau$'s matched to reco $\tau$'s"
    make_summary_plot "mass" "genTau_mass" "genTauMatched_mass" "Eff_vs_mass" "Efficiency_mass" 'Simulated $\tau$ $mass$' "${STEP} Tau Efficiency" "0,2" "0,1.2" "" "Efficiency" "Gen $\tau$'s" "Gen $\tau$'s matched to reco $\tau$'s"

    make_summary_plot "pt" "recoTau_pt" "recoTauMatched_pt" "Fake_vs_pt" "FakeRate_pt" '$\tau$ $p_T$ [GeV]' "${STEP} Tau Fake Rate" "0,400" "0,1.2" "2" "Fake rate" "Reco $\tau$'s" "Reco $\tau$'s matched to gen $\tau$'s" "--inverted"
    make_summary_plot "eta" "recoTau_eta" "recoTauMatched_eta" "Fake_vs_eta" "FakeRate_eta" '$\tau$ $\eta$' "${STEP} Tau Fake Rate" "-2.5,2.5" "0,1.2" "2" "Fake rate" "Reco $\tau$'s" "Reco $\tau$'s matched to gen $\tau$'s" "--inverted"
    make_summary_plot "phi" "recoTau_phi" "recoTauMatched_phi" "Fake_vs_phi" "FakeRate_phi" '$\tau$ $\phi$' "${STEP} Tau Fake Rate" "" "0,1.2" "2" "Fake rate" "Reco $\tau$'s" "Reco $\tau$'s matched to gen $\tau$'s" "--inverted"
    make_summary_plot "mass" "recoTau_mass" "recoTauMatched_mass" "Fake_vs_mass" "FakeRate_mass" '$\tau$ $mass$' "${STEP} Tau Fake Rate" "0,2" "0,1.2" "2" "Fake rate" "Reco $\tau$'s" "Reco $\tau$'s matched to gen $\tau$'s" "--inverted"

    make_summary_plot "pt" "genTau_pt" "genTauMultiMatched_pt" "Split_vs_pt" "SplitRate_pt" 'Simulated $\tau$ $p_T$ [GeV]' "${STEP} Tau Split Rate" "0,400" "0,1.2" "2" "Split rate" "Gen $\tau$'s" "Gen $\tau$'s matched to multiple reco $\tau$'s"
    make_summary_plot "eta" "genTau_eta" "genTauMultiMatched_eta" "Split_vs_eta" "SplitRate_eta" 'Simulated $\tau$ $\eta$' "${STEP} Tau Split Rate" "-2.5,2.5" "0,1.2" "2" "Split rate" "Gen $\tau$'s" "Gen $\tau$'s matched to multiple reco $\tau$'s"
    make_summary_plot "phi" "genTau_phi" "genTauMultiMatched_phi" "Split_vs_phi" "SplitRate_phi" 'Simulated $\tau$ $\phi$' "${STEP} Tau Split Rate" "" "0,1.2" "2" "Split rate" "Gen $\tau$'s" "Gen $\tau$'s matched to multiple reco $\tau$'s"
    make_summary_plot "mass" "genTau_mass" "genTauMultiMatched_mass" "Split_vs_mass" "SplitRate_mass" 'Simulated $\tau$ $mass$' "${STEP} Tau Split Rate" "0,2" "0,1.2" "2" "Split rate" "Gen $\tau$'s" "Gen $\tau$'s matched to multiple reco $\tau$'s"

    make_summary_plot "pt" "recoTau_pt" "recoTauMultiMatched_pt" "Dup_vs_pt" "DupRate_pt" '$\tau$ $p_T$ [GeV]' "${STEP} Tau Duplicate Rate" "0,400" "0,1.2" "2" "Duplicate rate" "Reco $\tau$'s" "Reco $\tau$'s matched to multiple gen $\tau$'s"
    make_summary_plot "eta" "recoTau_eta" "recoTauMultiMatched_eta" "Dup_vs_eta" "DupRate_eta" '$\tau$ $\eta$' "${STEP} Tau Duplicate Rate" "-2.5,2.5" "0,1.2" "2" "Duplicate rate" "Reco $\tau$'s" "Reco $\tau$'s matched to multiple gen $\tau$'s"
    make_summary_plot "phi" "recoTau_phi" "recoTauMultiMatched_phi" "Dup_vs_phi" "DupRate_phi" '$\tau$ $\phi$' "${STEP} Tau Duplicate Rate" "" "0,1.2" "2" "Duplicate rate" "Reco $\tau$'s" "Reco $\tau$'s matched to multiple gen $\tau$'s"
    make_summary_plot "mass" "recoTau_mass" "recoTauMultiMatched_mass" "Dup_vs_mass" "DupRate_mass" '$\tau$ $mass$' "${STEP} Tau Duplicate Rate" "0,2" "0,1.2" "2" "Duplicate rate" "Reco $\tau$'s" "Reco $\tau$'s matched to multiple gen $\tau$'s"

    echo
    echo "Done with ${step_upper}"
done

echo
echo "All steps complete."

make_hlt_vs_reco_plot() {
    local variable="$1"
    local hist="$2"
    local name="$3"
    local xlabel="$4"
    local ylabel="$5"
    local xlim="${6}"
    local ylim="${7}"
    local ylim_ratio="${8}"
    local rebin="${9}"
    local inverted="${10:-0}"

    local hlt="${SELECTED_DIR}/${hist}"
    local reco="${SELECTED_DIR}/${hist}"

    HIST_HLT="DQMData/Run 1/HLT/Run summary/Tau/TauValidation/${SUB_DIR}/${hist}"
    HIST_RECO="DQMData/Run 1/Tau/Run summary/TauValidation/${SUB_DIR}/${hist}"

    local cmd=(
        python3 "$MAKE_COMPARISON"
        --files "${DQM_FILE},${DQM_FILE}"
        --hists "$HIST_HLT,$HIST_RECO"
        --labels "HLT,RECO"
        --xlabel "$xlabel"
        --ylabel "$ylabel"
        --leg-title "Tau Performance"
        --energy-text "$ENERGY_TEXT"
        --odir "$OUTDIR_COMPARISON"
        --name "$name"
    )

    if [ -n "$xlim" ]; then
        cmd+=("--xlim=${xlim}")
    fi

    if [ -n "$ylim" ]; then
        cmd+=("--ylim=${ylim}")
    fi

    if [ -n "$ylim_ratio" ]; then
        cmd+=("--ylim-ratio=${ylim_ratio}")
    fi

    if [ -n "$rebin" ]; then
        cmd+=("--rebin=${rebin}")
    fi

    if [ "$inverted" -eq 1 ]; then
        cmd+=(--inverted)
    fi

    run_cmd "${cmd[@]}"
}

# If the user selected both steps, make comparison plots
if [[ " ${STEPS[*]} " == *" HLT "* ]] && [[ " ${STEPS[*]} " == *" RECO "* ]]; then
    echo
    echo "Making HLT vs RECO comparison plots"

    mkdir -p "$OUTDIR_COMPARISON"

    make_hlt_vs_reco_plot "pt" "Eff_vs_pt" "Eff_vs_pt_HLT_vs_RECO_comparison" 'Simulated $\tau$ $p_T$ [GeV]' "Efficiency" "0,400" "0,1.3" "0,2" "$PT_REBIN"
    make_hlt_vs_reco_plot "eta" "Eff_vs_eta" "Eff_vs_eta_HLT_vs_RECO_comparison" 'Simulated $\tau$ $\eta$' "Efficiency" "-2.5,2.5" "0,1.3" "0,2" "$ETA_REBIN"
    make_hlt_vs_reco_plot "phi" "Eff_vs_phi" "Eff_vs_phi_HLT_vs_RECO_comparison" 'Simulated $\tau$ $\phi$' "Efficiency" "" "0,1.3" "0,2" "$PHI_REBIN"
    make_hlt_vs_reco_plot "mass" "Eff_vs_mass" "Eff_vs_mass_HLT_vs_RECO_comparison" 'Simulated $\tau$ $mass$' "Efficiency" "0,2" "0,1.3" "0,2" "2"

    make_hlt_vs_reco_plot "pt" "Fake_vs_pt" "Fake_vs_pt_HLT_vs_RECO_comparison" '$\tau$ $p_T$ [GeV]' "Fake rate" "0,400" "0,1.3" "0,2" "$PT_REBIN" "1"
    make_hlt_vs_reco_plot "eta" "Fake_vs_eta" "Fake_vs_eta_HLT_vs_RECO_comparison" '$\tau$ $\eta$' "Fake rate" "-2.5,2.5" "0,1.3" "0,2" "$ETA_REBIN" "1"
    make_hlt_vs_reco_plot "phi" "Fake_vs_phi" "Fake_vs_phi_HLT_vs_RECO_comparison" '$\tau$ $\phi$' "Fake rate" "" "0,1.3" "0,2" "$PHI_REBIN" "1"
    make_hlt_vs_reco_plot "mass" "Fake_vs_mass" "Fake_vs_mass_HLT_vs_RECO_comparison" '$\tau$ $mass$' "Fake rate" "0,2" "0,1.3" "0,2" "2" "1"

    make_hlt_vs_reco_plot "pt" "Split_vs_pt" "Split_vs_pt_HLT_vs_RECO_comparison" 'Simulated $\tau$ $p_T$ [GeV]' "Split rate" "0,400" "0,1.3" "0,2" "$PT_REBIN"
    make_hlt_vs_reco_plot "eta" "Split_vs_eta" "Split_vs_eta_HLT_vs_RECO_comparison" 'Simulated $\tau$ $\eta$' "Split rate" "-2.5,2.5" "0,1.3" "0,2" "$ETA_REBIN"
    make_hlt_vs_reco_plot "phi" "Split_vs_phi" "Split_vs_phi_HLT_vs_RECO_comparison" 'Simulated $\tau$ $\phi$' "Split rate" "" "0,1.3" "0,2" "$PHI_REBIN"
    make_hlt_vs_reco_plot "mass" "Split_vs_mass" "Split_vs_mass_HLT_vs_RECO_comparison" 'Simulated $\tau$ $mass$' "Split rate" "0,2" "0,1.3" "0,2" "2"

    make_hlt_vs_reco_plot "pt" "Dup_vs_pt" "Dup_vs_pt_HLT_vs_RECO_comparison" '$\tau$ $p_T$ [GeV]' "Duplicate rate" "0,400" "0,1.3" "0,2" "$PT_REBIN"
    make_hlt_vs_reco_plot "eta" "Dup_vs_eta" "Dup_vs_eta_HLT_vs_RECO_comparison" '$\tau$ $\eta$' "Duplicate rate" "-2.5,2.5" "0,1.3" "0,2" "$ETA_REBIN"
    make_hlt_vs_reco_plot "phi" "Dup_vs_phi" "Dup_vs_phi_HLT_vs_RECO_comparison" '$\tau$ $\phi$' "Duplicate rate" "" "0,1.3" "0,2" "$PHI_REBIN"
    make_hlt_vs_reco_plot "mass" "Dup_vs_mass" "Dup_vs_mass_HLT_vs_RECO_comparison" '$\tau$ $mass$' "Duplicate rate" "0,2" "0,1.3" "0,2" "2"

    echo
    echo "Comparison plots complete"
fi

echo
echo "All plots complete."
