#! /bin/bash

function die { cat 29690*/*.log; echo $1: status $2; exit $2; }

# --command="-s HARVESTING:@HGCalValidation" overwrites the default "-s" to run only HGCAL harvesting
# 29690.0 is SingleGamma eta1p7to2p7 D110
runTheMatrix.py -w upgrade -l 29690.0 --startFrom HARVESTING --maxSteps 4 --recycle das:/RelValTTbar_14TeV/CMSSW_15_1_0-150X_mcRun4_realistic_v1_STD_RecycledGS_Run4D110_noPU-v2/DQMIO --command="-s HARVESTING:@HGCalValidation" || die "Could not run HARVESTING" $?

(makeHGCalValidationPlots.py --collection all --jobs 4 --ticlv 4 29690.0*/DQM*.root) || die "makeHGCalValidationPlots.py failed" $?
