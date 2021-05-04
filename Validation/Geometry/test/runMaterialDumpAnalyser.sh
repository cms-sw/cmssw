#!/bin/bash

set -x

# Check if CMSSW envs are setup
: ${CMSSW_BASE:?'You need to set CMSSW environemnt first.'}

# DEFAULTS

events=5000
geometry='Extended2016'

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

. runMaterialDumpFunctions

# GEN-SIM goes first
if checkFile SingleMuPt10_pythia8_cfi_GEN_SIM.root ; then
  cmsDriver.py SingleMuPt10_pythia8_cfi \
-s GEN,SIM \
--conditions auto:run2_mc \
-n ${events} \
--era Run2_2016 \
--eventcontent FEVTDEBUG \
--datatier GEN-SIM \
--beamspot NoSmear \
--nThreads=4 \
--fileout file:SingleMuPt10_pythia8_cfi_GEN_SIM.root \
--python_filename SingleMuPt10_pythia8_cfi_GEN_SIM.py > SingleMuPt10_pythia8_cfi_GEN_SIM.log 2>&1

    if [ $? -ne 0 ]; then
      echo "Error executing the GEN-SIM step, aborting."
      exit 1
    fi
fi

# DIGI comes next

if checkFile SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT.root ; then
  cmsDriver.py step2  \
-s DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval2016 \
--conditions auto:run2_mc \
-n -1 \
--era Run2_2016 \
--eventcontent FEVTDEBUGHLT \
--datatier GEN-SIM-DIGI-RAW \
--nThreads=4 \
--filein file:SingleMuPt10_pythia8_cfi_GEN_SIM.root  \
--fileout file:SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT.root \
--python_filename SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT.py > SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT.log 2>&1

    if [ $? -ne 0 ]; then
      echo "Error executing the DIGI step, aborting."
      exit 1
    fi
fi

# Reco and special customization
if checkFile SingleMuPt10_step3_RECO_DQM.root ; then
  cmsDriver.py step3  \
-s RAW2DIGI,L1Reco,RECO:reconstruction_trackingOnly,VALIDATION:@trackingOnlyValidation,DQM:@trackingOnlyDQM \
--conditions auto:run2_mc \
-n -1 \
--era Run2_2016 \
--eventcontent RECOSIM,DQM \
--datatier GEN-SIM-RECO,DQMIO \
--nThreads=4 \
--filein file:SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT.root  \
--fileout file:SingleMuPt10_step3_RECO_DQM.root \
--python_filename SingleMuPt10_step2_RECO_DQM.py > SingleMuPt10_step3_RECO_DQM.log 2>&1

    if [ $? -ne 0 ]; then
      echo "Error executing the RECO step, aborting."
      exit 1
    fi
fi

# HARVESTING
if checkFile DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root ; then
  cmsDriver.py step4  \
-s HARVESTING:@trackingOnlyValidation+@trackingOnlyDQM  \
--conditions auto:run2_mc \
-n -1   \
--era Run2_2016  \
--scenario pp  \
--filetype DQM  \
--mc  \
--filein file:SingleMuPt10_step3_RECO_DQM_inDQM.root  \
--python_filename SingleMuPt10_step4_HARVESTING.py > SingleMuPt10_step4_HARVESTING.log 2>&1

    if [ $? -ne 0 ]; then
      echo "Error executing the HARVESTING step, aborting."
      exit 1
    fi
fi

# Neutrino Particle gun

if checkFile single_neutrino_random.root ; then
  cmsRun ../python/single_neutrino_cfg.py
  if [ $? -ne 0 ]; then
    echo "Error generating single neutrino gun, aborting."
    exit 1
  fi
  if [ ! -e Images ]; then
    mkdir Images
  fi
fi

# Make material map for each subdetector from simulation

for t in BeamPipe Tracker PixBar PixFwdMinus PixFwdPlus TIB TOB TIDB TIDF TEC TkStrct InnerServices; do
  if [ ! -e matbdg_${t}.root ]; then
    python runP_Tracker.py geom=${geometry} label=$t >& /dev/null &
  fi
done

waitPendingJobs

# Always run the comparison at this stage, since you are guaranteed that all the ingredients are there

for t in BeamPipe Tracker PixBar PixFwdMinus PixFwdPlus TIB TOB TIDB TIDF TEC TkStrct InnerServices; do
  python MaterialBudget.py -s -d ${t}
  if [ $? -ne 0 ]; then
    echo "Error while producing simulation material for ${t}, aborting"
    exit 1
  fi
done

if [ ! -e Images ]; then
  mkdir Figures
fi
python MaterialBudget.py -c -r DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root -l Run2Detector > MaterialBudget_Simul_vs_Reco_Run2Detector.log 2>&1
