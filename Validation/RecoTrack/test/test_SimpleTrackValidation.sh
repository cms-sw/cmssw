#!/bin/bash
function die { echo $1: status $2; exit $2; }

echo "TESTING SimpleTrackValidation"

TEST_FILE="/store/relval/CMSSW_15_1_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_STD_Run4D110_PU-v1/2590000/ff561068-7e39-413b-8906-9f63338bf01c.root"

cmsRun $SCRAM_TEST_PATH/SimpleTrackValidation_cfg.py inputFiles=$TEST_FILE || die "Failure running test SimpleTrackValidation" $?
