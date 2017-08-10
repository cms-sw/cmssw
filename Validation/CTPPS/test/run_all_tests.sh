#!/bin/bash

here="${PWD}"
validation_path="${CMSSW_BASE}/src/Validation/CTPPS/test"

function RunOne()
{
	local tag="$1"
    local out_dir="$2"
    local out_logs="${here}/results/${out_dir}"
    mkdir -p "${out_logs}"
	cmsRun "${validation_path}/${out_dir}/${tag}_cfg.py" &> "${out_logs}/${tag}.log" &
}

RunOne "plot_optical_functions" "optics_parametrisation_validation"
RunOne "test_acceptance_shape" "fast_simu_validation"
RunOne "test_reco_simu_diff_with_det_sm" "fast_simu_validation"
RunOne "test_reco_simu_diff_without_det_sm" "fast_simu_validation"
#RunOne "test_y_vs_x_profile" "test_reco_simu_diff_without_det_sm" # already embedded in the previous test
RunOne "test_long_extrapolation" "fast_simu_validation"
RunOne "test" "fast_simu_with_phys_generator/qgsjet/global"
RunOne "test" "fast_simu_with_phys_generator/qgsjet/relative_to_beam"
