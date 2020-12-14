#!/bin/bash

source /afs/cern.ch/cms/cmsset_default.sh

mkdir test_tauDQM
cd test_tauDQM
cmsrel CMSSW_11_0_0
cd CMSSW_11_0_0/src
cmsenv
git cms-addpkg Validation/RecoTau
git cms-addpkg Configuration/DataProcessing
git cms-addpkg Configuration/StandardSequences
scram b -j 8
git remote add ece_cmssw https://github.com/easilar/cmssw.git
git pull ece_cmssw CMSSW_11_0_X
scram b -j 8
cmsenv
voms-proxy-init --rfc --voms cms
cd Validation/RecoTau/test/
cmsRun config.py
