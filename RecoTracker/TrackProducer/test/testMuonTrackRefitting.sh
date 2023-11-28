#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo "TESTING Refitting from Analysis Data-Tier codes ..."

# test AOD
echo "TESTING refitting from AOD ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/refitFromAOD.py maxEvents=100 inputFiles=/store/mc/RunIISummer20UL16RECO/DYJetsToMuMu_M-50_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/AODSIM/106X_mcRun2_asymptotic_v13-v2/00000/1D4FBF4B-7B8D-A04F-A108-B1BEB60558FA.root || die "Failure refitting from AOD" $?

# test MINIAOD
echo "TESTING refitting from MINIAOD ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/refitFromMINIAOD.py maxEvents=100 inputFiles=/store/mc/RunIISummer20UL16MiniAOD/DYJetsToMuMu_M-50_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/MINIAODSIM/106X_mcRun2_asymptotic_v13-v2/00000/01916D91-A314-D947-A420-388891D62FEA.root || die "Failure refitting from MINIAOD" $?
