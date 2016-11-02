#!/bin/bash
#cmsenv
source /cvmfs/cms.cern.ch/crab3/crab.sh
#voms-proxy-init -voms cms
#for i in `cat datasets_MCRUN2_25ns.txt` ; do
for i in `cat $1` ; do
  export DATASET=$i
  LFNBASE=/store/user/jpata/VHBBHeppyV13/ SITE=T2_EE_Estonia crab submit heppy_crab_config_env.py
  #LFNBASE=/store/t3groups/ethz-higgs/run2/VHBBHeppyV13/ SITE=T3_CH_PSI python heppy_crab_config_env.py
done 
