#!/bin/bash

set -x

# Check if CMSSW envs are setup
: ${CMSSW_BASE:?'You need to set CMSSW environemnt first.'}

# GEN-SIM goes first
cmsDriver.py SingleMuPt10_pythia8_cfi \
-s GEN,SIM \
--conditions auto:phase1_2017_realistic \
-n 500 \
--era Run2_2017_NewFPix \
--eventcontent FEVTDEBUG \
--datatier GEN-SIM \
--geometry Extended2017NewFPix  \
--beamspot NoSmear \
--fileout file:SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseI.root \
--python_filename SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseI.py > SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseI.log 2>&1

if [ $? -ne 0 ]; then
  echo "Error executing the GEN-SIM step, aborting."
  exit 1
fi

# DIGI comes next

cmsDriver.py step2  \
-s DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@fake \
--conditions auto:phase1_2017_realistic \
-n -1 \
--era Run2_2017_NewFPix \
--eventcontent FEVTDEBUGHLT \
--datatier GEN-SIM-DIGI-RAW \
--geometry Extended2017NewFPix \
--filein file:SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseI.root  \
--fileout file:SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseI.root \
--python_filename SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseI.py > SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseI.log 2>&1

if [ $? -ne 0 ]; then
  echo "Error executing the DIGI step, aborting."
  exit 1
fi

# Reco and special customization

cmsDriver.py step3  \
-s RAW2DIGI,L1Reco,RECO:reconstruction_trackingOnly,VALIDATION:@trackingOnlyValidation,DQM:@trackingOnlyDQM \
--conditions auto:phase1_2017_realistic \
-n -1 \
--era Run2_2017_NewFPix \
--eventcontent RECOSIM,DQM \
--datatier GEN-SIM-RECO,DQMIO \
--geometry Extended2017NewFPix \
--filein file:SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseI.root  \
--fileout file:SingleMuPt10_step3_RECO_DQM_PhaseI.root \
--python_filename SingleMuPt10_step2_RECO_DQM_PhaseI.py \
--customise Validation/Geometry/customiseForDumpMaterialAnalyser.customiseForMaterialAnalyser > SingleMuPt10_step3_RECO_DQM_PhaseI.log 2>&1

if [ $? -ne 0 ]; then
  echo "Error executing the RECO step, aborting."
  exit 1
fi

# HARVESTING

cmsDriver.py step4  \
-s HARVESTING:@trackingOnlyValidation+@trackingOnlyDQM  \
--conditions auto:phase1_2017_realistic \
-n -1   \
--era Run2_2017_NewFPix  \
--scenario pp  \
--filetype DQM  \
--geometry Extended2017NewFPix  \
--mc  \
--filein file:SingleMuPt10_step3_RECO_DQM_PhaseI_inDQM.root  \
--python_filename SingleMuPt10_step4_HARVESTING_PhaseI.py > SingleMuPt10_step4_HARVESTING_PhaseI.log 2>&1

if [ $? -ne 0 ]; then
  echo "Error executing the HARVESTING step, aborting."
  exit 1
fi
