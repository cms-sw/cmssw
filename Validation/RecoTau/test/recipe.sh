#!/bin/bash

source /afs/cern.ch/cms/cmsset_default.sh

mkdir test_tauDQM
cd test_tauDQM
cmsrel CMSSW_12_4_0_pre1
cd CMSSW_12_4_0_pre1/src
cmsenv
git cms-merge-topic cms-tau-pog:CMSSW_12_4_X-tau-pog_TauDQM
scram b -j 8

voms-proxy-init --rfc --voms cms
cd Validation/RecoTau/test/
cmsRun config.py
