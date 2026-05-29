#!/usr/bin/env bash

: <<'COMMENT'

Usage: run_tau_comparison_plots.sh [ID|WP|DeltaR] [HLT|RECO]

  ID     - scan on the ID raw values (default)
  WP     - scan on the working point
  DeltaR - scan on the DeltaR matching

  HLT   - use HLT DQM path and labels (default)
  RECO  - use RECO DQM path and labels

It assumes input DQM file, for DeltaR comparison the user needs to activate the hlt/recoTauValidation_deltaR sequence (de-activated by default) 
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

mode="ID"
if [ "$#" -gt 0 ]; then
    mode="$1"
fi

step="HLT"
if [ "$#" -gt 1 ]; then
    step="$2"
fi

mode_upper="${mode^^}"
case "$mode_upper" in
    ID)
        PREFIX="RawId"
        SUFFIX=""
        DIRS_LIST=(
            ""
            "CutID_VSjet0p50"
            "CutID_VSjet0p70"
            "CutID_VSjet0p90"
            "CutID_VSjet0p95"
            "CutID_VSjet0p99"
        )
        LABELS_LIST=(
            'No ID'
            'IDvsJet > 0.50'
            'IDvsJet > 0.70'
            'IDvsJet > 0.90'
            'IDvsJet > 0.95'
            'IDvsJet > 0.99'
        )
        ;;
    WP)
        PREFIX="WP"
        SUFFIX=""
        DIRS_LIST=(
            ""
            "CutWP_VSjet0"
            "CutWP_VSjet1"
            "CutWP_VSjet2"
            "CutWP_VSjet3"
        )
        LABELS_LIST=(
            'No ID'
            'WPvsJet > 0'
            'WPvsJet > 1'
            'WPvsJet > 2'
            'WPvsJet > 3'
        )
        ;;
    DELTAR)
        PREFIX="DeltaR"
        SUFFIX="_DeltaR"
        DIRS_LIST=(
            "DeltaR0p3"
            "DeltaR0p25"
            "DeltaR0p2"
            "DeltaR0p15"
            "DeltaR0p1"
        )
        LABELS_LIST=(
            '$\Delta R=0.3$'
            '$\Delta R=0.25$'
            '$\Delta R=0.2$'
            '$\Delta R=0.15$'
            '$\Delta R=0.1$'
        )
        ;;
    *)
        echo "Invalid mode: $mode"
        ;;
esac

step_upper="${step^^}"
case "$step_upper" in
    HLT)
        STEP="HLT"
        BASE_DIR="DQMData/Run 1/HLT/Run summary/Tau/TauValidation${SUFFIX}"
        ;;
    RECO)
        STEP="Reco"
        BASE_DIR="DQMData/Run 1/Tau/Run summary/TauValidation${SUFFIX}"
        ;;
    *)
        echo "Invalid step: $step"
        ;;
esac

# Change label according to the input file used
ENERGY_TEXT="Ten Tau (150 PU) | 14 TeV"

DQM_FILE="DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root"
SCRIPT_DIR="${CMSSW_BASE}/src/Validation/RecoTau/scripts"

MAKE_COMPARISON="${SCRIPT_DIR}/makeComparisonPlots.py"
MAKE_TAU_VALIDATION="${SCRIPT_DIR}/makeTauValidationPlots.py"

OUTDIR_COMPARISON="TauValidationPlots/Comparison_${mode_upper}_${step_upper}"

PT_REBIN="0,5,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,220,240,260,280,300"
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

make_comparison_plot() {
    local hist_base="$1"
    local name="$2"
    local xlabel="$3"
    local ylabel="$4"
    local xlim="$5"
    local ylim="$6"
    local ylim_ratio="$7"
    local rebin="$8"
    local inverted="${9:-0}"

    local files=()
    local hists=()
    local labels=()

    for i in "${!DIRS_LIST[@]}"; do
        local hist="${BASE_DIR}/${DIRS_LIST[$i]}/${hist_base}"

        if has_hist "$hist"; then
            files+=("$DQM_FILE")
            hists+=("$hist")
            labels+=("${LABELS_LIST[$i]}")
        else
            echo "Skipping missing histogram: $hist"
        fi
    done

    if [ "${#hists[@]}" -eq 0 ]; then
        echo "No valid histograms for ${hist_base}. Skipping."
        return
    fi

    local cmd=(
        python3 "$MAKE_COMPARISON"
        --files "$(join_by_comma "${files[@]}")"
        --hists "$(join_by_comma "${hists[@]}")"
        --labels "$(join_by_comma "${labels[@]}")"
        --xlabel "$xlabel"
        --ylabel "$ylabel"
        --leg-title "$STEP Tau Performance"
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

mkdir -p "$OUTDIR_COMPARISON"

echo "Input file:"
echo "$DQM_FILE"

echo
echo "Making comparison plots"

make_comparison_plot "Eff_vs_pt" "Eff_vs_pt_${PREFIX}_comparison" 'Simulated $\tau$ $p_T$ [GeV]' "Efficiency" "0,300" "0,1.4" "0,2" "$PT_REBIN"
make_comparison_plot "Eff_vs_mass" "Eff_vs_mass_${PREFIX}_comparison" 'Simulated $\tau$ mass [GeV]' "Efficiency" "0,2" "0,1.4" "0,2" "2"
make_comparison_plot "Eff_vs_eta" "Eff_vs_eta_${PREFIX}_comparison" 'Simulated $\tau$ $\eta$' "Efficiency" "-2.5,2.5" "0,1.4" "0.5,1.5" "$ETA_REBIN"
make_comparison_plot "Eff_vs_phi" "Eff_vs_phi_${PREFIX}_comparison" 'Simulated $\tau$ $\phi$' "Efficiency" "" "0,1.4" "0.5,1.5" "$PHI_REBIN"

make_comparison_plot "Fake_vs_pt" "Fake_vs_pt_${PREFIX}_comparison" '$\tau$ $p_T$ [GeV]' "Fake rate" "0,300" "0,1.4" "0.5,1.5" "$PT_REBIN" 1
make_comparison_plot "Fake_vs_mass" "Fake_vs_mass_${PREFIX}_comparison" '$\tau$ mass [GeV]' "Fake rate" "0,2" "0,1.4" "0.5,1.5" "2" 1
make_comparison_plot "Fake_vs_eta" "Fake_vs_eta_${PREFIX}_comparison" '$\tau$ $\eta$' "Fake rate" "-2.5,2.5" "0,1.4" "0.5,1.5" "$ETA_REBIN" 1
make_comparison_plot "Fake_vs_phi" "Fake_vs_phi_${PREFIX}_comparison" '$\tau$ $\phi$' "Fake rate" "" "0,1.4" "0.5,1.5" "$PHI_REBIN" 1

make_comparison_plot "Dup_vs_pt" "Dup_vs_pt_${PREFIX}_comparison" '$\tau$ $p_T$ [GeV]' "Duplicate rate" "0,300" "0,1" "0,2" "$PT_REBIN"
make_comparison_plot "Dup_vs_mass" "Dup_vs_mass_${PREFIX}_comparison" '$\tau$ $mass$ [GeV]' "Duplicate rate" "0,2" "0,1" "0,2" "2"
make_comparison_plot "Dup_vs_eta" "Dup_vs_eta_${PREFIX}_comparison" '$\tau$ $\eta$' "Duplicate rate" "-2.5,2.5" "0,0.1" "0,2" "$ETA_REBIN"
make_comparison_plot "Dup_vs_phi" "Dup_vs_phi_${PREFIX}_comparison" '$\tau$ $\phi$' "Duplicate rate" "" "0,0.1" "0,2" "$PHI_REBIN"

make_comparison_plot "Split_vs_pt" "Split_vs_pt_${PREFIX}_comparison" 'Simulated $\tau$ $p_T$ [GeV]' "Split rate" "0,300" "0,1" "0,2" "$PT_REBIN"
make_comparison_plot "Split_vs_mass" "Split_vs_mass_${PREFIX}_comparison" 'Simulated $\tau$ $mass$ [GeV]' "Split rate" "0,2" "0,1" "0,2" "2"
make_comparison_plot "Split_vs_eta" "Split_vs_eta_${PREFIX}_comparison" 'Simulated $\tau$ $\eta$' "Split rate" "-2.5,2.5" "0,1" "0,2" "$ETA_REBIN"
make_comparison_plot "Split_vs_phi" "Split_vs_phi_${PREFIX}_comparison" 'Simulated $\tau$ $\phi$' "Split rate" "" "0,1" "0,2" "$PHI_REBIN"

make_comparison_plot "ResponsePt_RecoOverGen_vs_pt_Mean" "ScalePt_vs_pt_${PREFIX}_comparison" 'Simulated $\tau$ $p_T$ [GeV]' "$\langle p_T^{reco}/p_T^{gen} \rangle$" "0,300" "0,2" "0,2" "$PT_REBIN"
make_comparison_plot "ResponsePt_RecoOverGen_vs_mass_Mean" "ScalePt_vs_mass_${PREFIX}_comparison" 'Simulated $\tau$ $mass$ [GeV]' "$\langle p_T^{reco}/p_T^{gen} \rangle$" "0,2" "0,2" "0,2" "2"
make_comparison_plot "ResponsePt_RecoOverGen_vs_eta_Mean" "ScalePt_vs_eta_${PREFIX}_comparison" 'Simulated $\tau$ $\eta$' "$\langle p_T^{reco}/p_T^{gen} \rangle$" "-2.5,2.5" "0,2" "0,2" "$ETA_REBIN"
make_comparison_plot "ResponsePt_RecoOverGen_vs_phi_Mean" "ScalePt_vs_phi_${PREFIX}_comparison" 'Simulated $\tau$ $\phi$' "$\langle p_T^{reco}/p_T^{gen} \rangle$" "" "0,2" "0,2" "$PHI_REBIN"

make_comparison_plot "ResponseMass_RecoOverGen_vs_pt_Mean" "ScaleMass_vs_pt_${PREFIX}_comparison" 'Simulated $\tau$ $p_T$ [GeV]' "$\langle m^{reco}/m^{gen} \rangle$" "0,300" "0,2" "0,2" "$PT_REBIN"
make_comparison_plot "ResponseMass_RecoOverGen_vs_mass_Mean" "ScaleMass_vs_mass_${PREFIX}_comparison" 'Simulated $\tau$ $mass$ [GeV]' "$\langle m^{reco}/m^{gen} \rangle$" "0,2" "0,2" "0,2" "2"
make_comparison_plot "ResponseMass_RecoOverGen_vs_eta_Mean" "ScaleMass_vs_eta_${PREFIX}_comparison" 'Simulated $\tau$ $\eta$' "$\langle m^{reco}/m^{gen} \rangle$" "-2.5,2.5" "0,2" "0,2" "$ETA_REBIN"
make_comparison_plot "ResponseMass_RecoOverGen_vs_phi_Mean" "ScaleMass_vs_phi_${PREFIX}_comparison" 'Simulated $\tau$ $\phi$' "$\langle m^{reco}/m^{gen} \rangle$" "" "0,2" "0,2" "$PHI_REBIN"

make_resolution_comparison() {
    local base="$1"
    local name="$2"
    local xlabel="$3"
    local ylabel="$4"
    local xlim="$5"
    local ylim="$6"
    local ylim_ratio="$7"
    local rebin="$8"

    local files=()
    local mean_hists=()
    local sigma_hists=()
    local labels=()

    for i in "${!DIRS_LIST[@]}"; do
        local mean_hist="${BASE_DIR}/${DIRS_LIST[$i]}/${base}_Mean"
        local sigma_hist="${BASE_DIR}/${DIRS_LIST[$i]}/${base}_Sigma"

        if has_hist "$mean_hist" && has_hist "$sigma_hist"; then
            files+=("$DQM_FILE")
            mean_hists+=("$mean_hist")
            sigma_hists+=("$sigma_hist")
            labels+=("${LABELS_LIST[$i]}")
        else
            echo "Skipping missing response pair:"
            echo "  mean : $mean_hist"
            echo "  sigma: $sigma_hist"
        fi
    done

    if [ "${#mean_hists[@]}" -eq 0 ]; then
        echo "No valid response histograms for ${base}. Skipping."
        return
    fi

    local cmd=(
        python3 "$MAKE_TAU_VALIDATION"
        --mode response
        --files "$(join_by_comma "${files[@]}")"
        --mean-hists "$(join_by_comma "${mean_hists[@]}")"
        --sigma-hists "$(join_by_comma "${sigma_hists[@]}")"
        --labels "$(join_by_comma "${labels[@]}")"
        --xlabel "$xlabel"
        --ylabel "$ylabel"
        --energy-text "$ENERGY_TEXT"
        --odir "$OUTDIR_COMPARISON"
        --name "$name"
        --title "$name"
        --leg-title "$STEP Tau Performance"
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

    run_cmd "${cmd[@]}"
}

make_resolution_comparison "ResponsePt_RecoOverGen_vs_pt" "ResolutionPt_vs_pt_${PREFIX}_comparison" 'Simulated $\tau$ $p_T$ [GeV]' '$\sigma(p_T^{reco}/p_T^{gen}) / \langle p_T^{reco}/p_T^{gen} \rangle$' "0,300" "0,0.7" "0.5,1.5" "$PT_REBIN"
make_resolution_comparison "ResponsePt_RecoOverGen_vs_mass" "ResolutionPt_vs_mass_${PREFIX}_comparison" 'Simulated $\tau$ $mass$ [GeV]' '$\sigma(p_T^{reco}/p_T^{gen}) / \langle p_T^{reco}/p_T^{gen} \rangle$' "0,2" "0,0.7" "0.5,1.5" "2"
make_resolution_comparison "ResponsePt_RecoOverGen_vs_eta" "ResolutionPt_vs_eta_${PREFIX}_comparison" 'Simulated $\tau$ $\eta$' '$\sigma(p_T^{reco}/p_T^{gen}) / \langle p_T^{reco}/p_T^{gen} \rangle$' "-2.5,2.5" "0,0.7" "0.5,1.5" "$ETA_REBIN"
make_resolution_comparison "ResponsePt_RecoOverGen_vs_phi" "ResolutionPt_vs_phi_${PREFIX}_comparison" 'Simulated $\tau$ $\phi$' '$\sigma(p_T^{reco}/p_T^{gen}) / \langle p_T^{reco}/p_T^{gen} \rangle$' "" "0,0.7" "0.5,1.5" "$PHI_REBIN"

make_resolution_comparison "ResponseMass_RecoOverGen_vs_pt" "ResolutionMass_vs_pt_${PREFIX}_comparison" 'Simulated $\tau$ $p_T$ [GeV]' '$\sigma(m^{reco}/m^{gen}) / \langle m^{reco}/m^{gen} \rangle$' "0,300" "0,0.7" "0.5,1.5" "$PT_REBIN"
make_resolution_comparison "ResponseMass_RecoOverGen_vs_mass" "ResolutionMass_vs_mass_${PREFIX}_comparison" 'Simulated $\tau$ $mass$ [GeV]' '$\sigma(m^{reco}/m^{gen}) / \langle m^{reco}/m^{gen} \rangle$' "0,2" "0,0.7" "0.5,1.5" "2"
make_resolution_comparison "ResponseMass_RecoOverGen_vs_eta" "ResolutionMass_vs_eta_${PREFIX}_comparison" 'Simulated $\tau$ $\eta$' '$\sigma(m^{reco}/m^{gen}) / \langle m^{reco}/m^{gen} \rangle$' "-2.5,2.5" "0,0.7" "0.5,1.5" "$ETA_REBIN"
make_resolution_comparison "ResponseMass_RecoOverGen_vs_phi" "ResolutionMass_vs_phi_${PREFIX}_comparison" 'Simulated $\tau$ $\phi$' '$\sigma(m^{reco}/m^{gen}) / \langle m^{reco}/m^{gen} \rangle$' "" "0,0.7" "0.5,1.5" "$PHI_REBIN"

echo
echo "Done."