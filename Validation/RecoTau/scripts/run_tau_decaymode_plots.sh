#!/usr/bin/env bash

set -u

: <<'COMMENT'
Usage:
  run_tau_decaymode_plots.sh [HLT|RECO] [SUBDIR]

Examples:
  ./run_tau_decaymode_plots.sh HLT
  ./run_tau_decaymode_plots.sh RECO
  ./run_tau_decaymode_plots.sh HLT CutID_VSjet0p70
  ./run_tau_decaymode_plots.sh HLT CutWP_VSjet0

This script overlays tau validation quantities split by decay mode.

It assumes the harvested DQM file exists in the current directory:
  DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root

For DeltaR = 0.3, use the default TauValidation folder.
Do not use TauValidation_DeltaR unless you explicitly enabled the DeltaR scan sequence.
COMMENT

STEP_IN="${1:-HLT}"
SUB_DIR="${2:-}"

STEP_UPPER="${STEP_IN^^}"

case "$STEP_UPPER" in
    HLT)
        STEP="HLT"
        BASE_DIR="DQMData/Run 1/HLT/Run summary/Tau/TauValidation"
        ;;
    RECO)
        STEP="Reco"
        BASE_DIR="DQMData/Run 1/Tau/Run summary/TauValidation"
        ;;
    *)
        echo "Invalid step: $STEP_IN"
        echo "Use HLT or RECO"
        exit 1
        ;;
esac

DQM_FILE="DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root"

SCRIPT_DIR="${CMSSW_BASE}/src/Validation/RecoTau/scripts"
MAKE_COMPARISON="${SCRIPT_DIR}/makeComparisonPlots.py"
MAKE_TAU_VALIDATION="${SCRIPT_DIR}/makeTauValidationPlots.py"

ENERGY_TEXT="Ten Tau | 14 TeV"
LABEL_TEXT="${STEP} Tau validation, #Delta R = 0.3"

if [ -n "$SUB_DIR" ]; then
    SELECTED_DIR="${BASE_DIR}/${SUB_DIR}"
    OUTDIR="TauValidationPlots/DecayModes_${STEP_UPPER}_${SUB_DIR}"
    LABEL_TEXT="${LABEL_TEXT}, ${SUB_DIR}"
else
    SELECTED_DIR="${BASE_DIR}"
    OUTDIR="TauValidationPlots/DecayModes_${STEP_UPPER}_NoCut"
fi

PT_REBIN="0,5,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,220,240,260,280,300,340,380,400"
ETA_REBIN="-2.4,-2.0,-1.6,-1.2,-0.8,-0.4,0.0,0.4,0.8,1.2,1.6,2.0,2.4"
PHI_REBIN="-3.5,-2.8,-2.1,-1.4,-0.7,0.0,0.7,1.4,2.1,2.8,3.5"

GEN_DECAY_MODES=(
    "oneProng0Pi0"
    "oneProng1Pi0"
    "oneProng2Pi0"
    "oneProngOther"
    "threeProng0Pi0"
    "threeProng1Pi0"
    "threeProngOther"
    "rare"
)

GEN_DECAY_LABELS=(
    "1-prong 0 #pi^{0}"
    "1-prong 1 #pi^{0}"
    "1-prong 2 #pi^{0}"
    "1-prong other"
    "3-prong 0 #pi^{0}"
    "3-prong 1 #pi^{0}"
    "3-prong other"
    "rare"
)

RECO_DECAY_MODES=(
    "oneProng0Pi0"
    "oneProng1Pi0"
    "oneProng2Pi0"
    "oneProngOther"
    "threeProng0Pi0"
    "threeProng1Pi0"
    "threeProngOther"
    "rare"
    "unknown"
)

RECO_DECAY_LABELS=(
    "1-prong 0 #pi^{0}"
    "1-prong 1 #pi^{0}"
    "1-prong 2 #pi^{0}"
    "1-prong other"
    "3-prong 0 #pi^{0}"
    "3-prong 1 #pi^{0}"
    "3-prong other"
    "rare"
    "unknown"
)

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

mkdir -p "$OUTDIR"

echo "Input file:"
echo "$DQM_FILE"

echo
echo "Selected DQM directory:"
echo "$SELECTED_DIR"

echo
echo "Making decay-mode plots for ${STEP_UPPER}, DeltaR = 0.3"

# Efficiency: gen decay mode
make_decaymode_overlay "Eff" "pt"   "Eff_decayModes_vs_pt"   'GenVis $\tau$ $p_T$ [GeV]' "Efficiency" "0,400" "0,1.3" "$PT_REBIN" "gen"
make_decaymode_overlay "Eff" "eta"  "Eff_decayModes_vs_eta"  'GenVis $\tau$ $\eta$'      "Efficiency" "-2.5,2.5" "0,1.3" "$ETA_REBIN" "gen"
make_decaymode_overlay "Eff" "phi"  "Eff_decayModes_vs_phi"  'GenVis $\tau$ $\phi$'      "Efficiency" "" "0,1.3" "$PHI_REBIN" "gen"
make_decaymode_overlay "Eff" "mass" "Eff_decayModes_vs_mass" 'GenVis $\tau$ mass [GeV]'  "Efficiency" "0,2" "0,1.3" "2" "gen"

# Fake rate: reco decay mode
make_decaymode_overlay "Fake" "pt"   "Fake_decayModes_vs_pt"   '$\tau$ $p_T$ [GeV]' "Fake rate" "0,400" "0,1.3" "$PT_REBIN" "reco" 1
make_decaymode_overlay "Fake" "eta"  "Fake_decayModes_vs_eta"  '$\tau$ $\eta$'      "Fake rate" "-2.5,2.5" "0,1.3" "$ETA_REBIN" "reco" 1
make_decaymode_overlay "Fake" "phi"  "Fake_decayModes_vs_phi"  '$\tau$ $\phi$'      "Fake rate" "" "0,1.3" "$PHI_REBIN" "reco" 1
make_decaymode_overlay "Fake" "mass" "Fake_decayModes_vs_mass" '$\tau$ mass [GeV]'  "Fake rate" "0,2" "0,1.3" "2" "reco" 1

# Split rate: gen decay mode
make_decaymode_overlay "Split" "pt"   "Split_decayModes_vs_pt"   'GenVis $\tau$ $p_T$ [GeV]' "Split rate" "0,400" "0,1.0" "$PT_REBIN" "gen"
make_decaymode_overlay "Split" "eta"  "Split_decayModes_vs_eta"  'GenVis $\tau$ $\eta$'      "Split rate" "-2.5,2.5" "0,1.0" "$ETA_REBIN" "gen"
make_decaymode_overlay "Split" "phi"  "Split_decayModes_vs_phi"  'GenVis $\tau$ $\phi$'      "Split rate" "" "0,1.0" "$PHI_REBIN" "gen"
make_decaymode_overlay "Split" "mass" "Split_decayModes_vs_mass" 'GenVis $\tau$ mass [GeV]'  "Split rate" "0,2" "0,1.0" "2" "gen"

# Duplicate rate: reco decay mode
make_decaymode_overlay "Dup" "pt"   "Dup_decayModes_vs_pt"   '$\tau$ $p_T$ [GeV]' "Duplicate rate" "0,400" "0,1.0" "$PT_REBIN" "reco"
make_decaymode_overlay "Dup" "eta"  "Dup_decayModes_vs_eta"  '$\tau$ $\eta$'      "Duplicate rate" "-2.5,2.5" "0,0.2" "$ETA_REBIN" "reco"
make_decaymode_overlay "Dup" "phi"  "Dup_decayModes_vs_phi"  '$\tau$ $\phi$'      "Duplicate rate" "" "0,0.2" "$PHI_REBIN" "reco"
make_decaymode_overlay "Dup" "mass" "Dup_decayModes_vs_mass" '$\tau$ mass [GeV]'  "Duplicate rate" "0,2" "0,1.0" "2" "reco"

# Response means: gen decay mode
make_response_decaymode_overlay "ResponsePt" "pt"   "ResponsePt_decayModes_vs_pt"   'GenVis $\tau$ $p_T$ [GeV]' "$\langle p_T^{reco}/p_T^{gen} \rangle$" "0,400" "0,2" "$PT_REBIN"
make_response_decaymode_overlay "ResponsePt" "eta"  "ResponsePt_decayModes_vs_eta"  'GenVis $\tau$ $\eta$'      "$\langle p_T^{reco}/p_T^{gen} \rangle$" "-2.5,2.5" "0,2" "$ETA_REBIN"
make_response_decaymode_overlay "ResponsePt" "phi"  "ResponsePt_decayModes_vs_phi"  'GenVis $\tau$ $\phi$'      "$\langle p_T^{reco}/p_T^{gen} \rangle$" "" "0,2" "$PHI_REBIN"
make_response_decaymode_overlay "ResponsePt" "mass" "ResponsePt_decayModes_vs_mass" 'GenVis $\tau$ mass [GeV]'  "$\langle p_T^{reco}/p_T^{gen} \rangle$" "0,2" "0,2" "2"

make_response_decaymode_overlay "ResponseMass" "pt"   "ResponseMass_decayModes_vs_pt"   'GenVis $\tau$ $p_T$ [GeV]' "$\langle m^{reco}/m^{gen} \rangle$" "0,400" "0,2" "$PT_REBIN"
make_response_decaymode_overlay "ResponseMass" "eta"  "ResponseMass_decayModes_vs_eta"  'GenVis $\tau$ $\eta$'      "$\langle m^{reco}/m^{gen} \rangle$" "-2.5,2.5" "0,2" "$ETA_REBIN"
make_response_decaymode_overlay "ResponseMass" "phi"  "ResponseMass_decayModes_vs_phi"  'GenVis $\tau$ $\phi$'      "$\langle m^{reco}/m^{gen} \rangle$" "" "0,2" "$PHI_REBIN"
make_response_decaymode_overlay "ResponseMass" "mass" "ResponseMass_decayModes_vs_mass" 'GenVis $\tau$ mass [GeV]'  "$\langle m^{reco}/m^{gen} \rangle$" "0,2" "0,2" "2"

echo
echo "Done."