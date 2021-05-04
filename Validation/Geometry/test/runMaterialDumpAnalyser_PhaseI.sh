#!/bin/bash

set -x

# Check if CMSSW envs are setup
: ${CMSSW_BASE:?'You need to set CMSSW environemnt first.'}

# DEFAULTS

events=5000
geometry=Extended2017Plan1

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
if checkFile SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseI.root ; then
cmsDriver.py SingleMuPt10_pythia8_cfi \
-s GEN,SIM \
--conditions auto:phase1_2017_realistic \
-n ${events} \
--era Run2_2017 \
--eventcontent FEVTDEBUG \
--datatier GEN-SIM \
--geometry ${geometry}  \
--beamspot NoSmear \
--customise Validation/Geometry/customiseForDumpMaterialAnalyser_ForPhaseI.customiseForMaterialAnalyser_ForPhaseI \
--fileout file:SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseI.root \
--python_filename SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseI.py > SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseI.log 2>&1

    if [ $? -ne 0 ]; then
      echo "Error executing the GEN-SIM step, aborting."
      exit 1
    fi
fi

# DIGI comes next

if checkFile SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseI.root ; then
  cmsDriver.py step2  \
-s DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval2017 \
--conditions auto:phase1_2017_realistic \
-n -1 \
--era Run2_2017 \
--eventcontent FEVTDEBUGHLT \
--datatier GEN-SIM-DIGI-RAW \
--geometry ${geometry} \
--nThreads 6 \
--filein file:SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseI.root  \
--fileout file:SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseI.root \
--python_filename SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseI.py > SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseI.log 2>&1

    if [ $? -ne 0 ]; then
      echo "Error executing the DIGI step, aborting."
      exit 1
    fi
fi

# Reco and special customization

if checkFile SingleMuPt10_step3_RECO_DQM_PhaseI.root ; then
  cmsDriver.py step3  \
-s RAW2DIGI,L1Reco,RECO:reconstruction_trackingOnly,VALIDATION:@trackingOnlyValidation,DQM:@trackingOnlyDQM \
--conditions auto:phase1_2017_realistic \
-n -1 \
--era Run2_2017 \
--eventcontent RECOSIM,DQM \
--datatier GEN-SIM-RECO,DQMIO \
--geometry ${geometry} \
--nThreads 6 \
--filein file:SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseI.root  \
--fileout file:SingleMuPt10_step3_RECO_DQM_PhaseI.root \
--python_filename SingleMuPt10_step2_RECO_DQM_PhaseI.py  > SingleMuPt10_step3_RECO_DQM_PhaseI.log 2>&1

    if [ $? -ne 0 ]; then
      echo "Error executing the RECO step, aborting."
      exit 1
    fi
fi

# HARVESTING

if checkFile DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root ; then
  cmsDriver.py step4  \
-s HARVESTING:@trackingOnlyValidation+@trackingOnlyDQM  \
--conditions auto:phase1_2017_realistic \
-n -1   \
--era Run2_2017  \
--scenario pp  \
--filetype DQM  \
--geometry ${geometry} \
--mc  \
--filein file:SingleMuPt10_step3_RECO_DQM_PhaseI_inDQM.root  \
--python_filename SingleMuPt10_step4_HARVESTING_PhaseI.py > SingleMuPt10_step4_HARVESTING_PhaseI.log 2>&1

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

for t in TrackerSum Pixel Strip InnerTracker BeamPipe Tracker PixBar PixFwdMinus PixFwdPlus TIB TOB TIDB TIDF TEC TkStrct InnerServices; do
  python MaterialBudget.py -s -d ${t}
  if [ $? -ne 0 ]; then
    echo "Error while producing simulation material for ${t}, aborting"
    exit 1
  fi
done

python MaterialBudget.py -c -r DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root -l PhaseIDetector > MaterialBudget_Simul_vs_Reco_PhaseIDetector.log 2>&1
