#!/bin/bash

set -x

# Check if CMSSW envs are setup
: ${CMSSW_BASE:?'You need to set CMSSW environemnt first.'}

# DEFAULTS

events=5000

# ARGUMENT PARSING

while getopts ":n:" opt; do
  case $opt in
    n)
      echo "Generating $OPTARG events" >&1
      events=${OPTARG}
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# GEN-SIM goes first

cmsDriver.py SingleMuPt10_pythia8_cfi \
-s GEN,SIM \
--conditions auto:run2_mc \
-n ${events} \
--era Phase2C1 \
--eventcontent FEVTDEBUG \
--datatier GEN-SIM \
--beamspot NoSmear \
--customise SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023tilted \
--geometry Extended2023D1 \
--fileout file:SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseII.root \
--python_filename SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseII.py > SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseII.log 2>&1

if [ $? -ne 0 ]; then
  echo "Error executing the GEN-SIM step, aborting."
  exit 1
fi

# DIGI comes next

cmsDriver.py step2   \
-s DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@fake  \
--conditions auto:run2_mc \
-n -1  \
--era Phase2C1  \
--eventcontent FEVTDEBUGHLT \
--datatier GEN-SIM-DIGI-RAW  \
--customise SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023tilted  \
--geometry Extended2023D1  \
--filein file:SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseII.root  \
--fileout file:SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseII.root \
--python_filename SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseII.py > SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseII.log 2>&1

if [ $? -ne 0 ]; then
  echo "Error executing the DIGI step, aborting."
  exit 1
fi

# Reco and special customization

cmsDriver.py step3 \
-s RAW2DIGI,RECO:reconstruction_trackingOnly,VALIDATION:@trackingOnlyValidation,DQM:@trackingOnlyDQM  \
--conditions auto:run2_mc \
-n -1  \
--runUnscheduled \
--era Phase2C1  \
--eventcontent RECOSIM,DQM  \
--datatier GEN-SIM-RECO,DQMIO  \
--customise SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023tilted  \
--geometry Extended2023D1  \
--filein file:SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseII.root  \
--fileout file:SingleMuPt10_step3_RECO_DQM_PhaseII.root \
--python_filename SingleMuPt10_step2_RECO_DQM_PhaseII.py \
--customise Validation/Geometry/customiseForDumpMaterialAnalyser.customiseForMaterialAnalyser > SingleMuPt10_step3_RECO_DQM_PhaseII.log 2>&1

if [ $? -ne 0 ]; then
  echo "Error executing the RECO step, aborting."
  exit 1
fi

# HARVESTING

cmsDriver.py step4  \
-s HARVESTING:@trackingOnlyValidation+@trackingOnlyDQM  \
--conditions auto:run2_mc \
-n -1   \
--era Phase2C1  \
--scenario pp  \
--filetype DQM  \
--customise SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023tilted  \
--geometry Extended2023D1  \
--mc  \
--filein file:SingleMuPt10_step3_RECO_DQM_PhaseII_inDQM.root  \
--python_filename SingleMuPt10_step4_HARVESTING_PhaseII.py > SingleMuPt10_step4_HARVESTING_PhaseII.log 2>&1

if [ $? -ne 0 ]; then
  echo "Error executing the HARVESTING step, aborting."
  exit 1
fi

