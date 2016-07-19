#!/bin/bash
eval `scramv1 runtime -sh`
source /cvmfs/cms.cern.ch/crab3/crab.sh
#voms-proxy-init -voms cms
#for i in `cat datasets_MCRUN2_25ns.txt` ; do
for i in `cat $1` ; do
  export DATASET=$i 
  crab submit -c heppy_crab_config_env.py 
done 
