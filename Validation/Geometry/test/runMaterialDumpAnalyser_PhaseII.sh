#!/bin/bash

set -x

# Check if CMSSW envs are setup
: ${CMSSW_BASE:?'You need to set CMSSW environemnt first.'}

# DEFAULTS

events=5000
geometry=Extended2023D11

# ARGUMENT PARSING

while getopts ":n:g:" opt; do
  case $opt in
    n)
      echo "Generating $OPTARG events" >&1
      events=${OPTARG}
      ;;
    g)
      echo "Using $OPTARG geometry" >&1
      geometry=${OPTARG}
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
if checkFile SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseII.root ; then
  cmsDriver.py SingleMuPt10_pythia8_cfi \
-s GEN,SIM \
--conditions auto:phase2_realistic \
-n ${events} \
--era Phase2C2 \
--eventcontent FEVTDEBUG \
--datatier GEN-SIM \
--beamspot NoSmear \
--customise Validation/Geometry/customiseForDumpMaterialAnalyser_ForPhaseII.customiseForMaterialAnalyser_ForPhaseII \
--geometry ${geometry} \
--fileout file:SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseII.root \
--python_filename SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseII.py > SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseII.log 2>&1

  if [ $? -ne 0 ]; then
    echo "Error executing the GEN-SIM step, aborting."
    exit 1
  fi
fi

# DIGI comes next
if checkFile SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseII.root ; then
  cmsDriver.py step2   \
-s DIGI:pdigi_valid,L1,L1TrackTrigger,DIGI2RAW,HLT:@fake2  \
--conditions auto:phase2_realistic \
-n -1  \
--era Phase2C2  \
--eventcontent FEVTDEBUGHLT \
--datatier GEN-SIM-DIGI-RAW  \
--nThreads=6 \
--geometry ${geometry}  \
--filein file:SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseII.root  \
--fileout file:SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseII.root \
--python_filename SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseII.py > SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseII.log 2>&1

  if [ $? -ne 0 ]; then
    echo "Error executing the DIGI step, aborting."
    exit 1
  fi
fi

# Reco and special customization
if checkFile SingleMuPt10_step3_RECO_DQM_PhaseII.root ; then
  cmsDriver.py step3 \
-s RAW2DIGI,L1Reco,RECO,VALIDATION:@phase2Validation,DQM:@phase2 \
--conditions auto:phase2_realistic \
-n -1  \
--runUnscheduled \
--era Phase2C2  \
--eventcontent FEVTDEBUGHLT,DQM  \
--datatier GEN-SIM-RECO,DQMIO  \
--nThreads=6 \
--geometry ${geometry}  \
--filein file:SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseII.root  \
--fileout file:SingleMuPt10_step3_RECO_DQM_PhaseII.root \
--python_filename SingleMuPt10_step2_RECO_DQM_PhaseII.py > SingleMuPt10_step3_RECO_DQM_PhaseII.log 2>&1

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
--era Phase2C2  \
--scenario pp  \
--filetype DQM  \
--geometry ${geometry}  \
--mc  \
--filein file:SingleMuPt10_step3_RECO_DQM_PhaseII_inDQM.root  \
--python_filename SingleMuPt10_step4_HARVESTING_PhaseII.py > SingleMuPt10_step4_HARVESTING_PhaseII.log 2>&1

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

for t in BeamPipe Tracker Phase2PixelBarrel Phase2OTBarrel Phase2PixelEndcap Phase2OTForward; do
  if [ ! -e matbdg_${t}.root ]; then
    python runP_Tracker.py geom=${geometry} label=$t >& /dev/null &
  fi
done

waitPendingJobs

# Always run the comparison at this stage, since you are guaranteed that all the ingredients are there

for t in BeamPipe Tracker TrackerSumPhaseII Phase2PixelBarrel Phase2OTBarrel Phase2PixelEndcap Phase2OTForward; do
  python MaterialBudget.py -s -d ${t}
  if [ $? -ne 0 ]; then
    echo "Error while producing simulation material for ${t}, aborting"
    exit 1
  fi
done

python MaterialBudget.py -c -r DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root -l PhaseIIDetector > MaterialBudget_Simul_vs_Reco_PhaseII.log 2>&1

