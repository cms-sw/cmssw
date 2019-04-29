
# Quickstart

Run locally on lxplus
~~~
#set up the work area
# for lxplus with SLC7 (default since April 2019)
export SCRAM_ARCH=slc7_amd64_gcc700
# for SLC6 use 'slc6_amd64_gcc700' instead above
cmsrel CMSSW_10_5_0_pre1
cd CMSSW_10_5_0_pre1
cmsenv

#get the code
git cms-checkout-topic jpata:pfvalidation-10_5_0_pre1-master
scram b -j4
cd $CMSSW_BASE/src/Validation/RecoParticleFlow

# Activate reading files from remote locations (needed at lxplus at least)
voms-proxy-init -voms cms

#RECO step, about 30 minutes
make QCD_reco

#DQM step, a few minutes
make QCD_dqm

#Do postprocessing on the DQM histograms
make QCD_post

# Repeat for QCDPU, ZMM and MinBias (make QCDPU_reco etc.) or edit the 'make plots' part of
# Makefile for successfully running 'make plots'

#Do final HTML plots
make plots

# The 'plots' directory can be viewed from a web browser once it is moved to e.g. /afs/cern.ch/user/f/foo/www/.
# In this case the URL for the directory is 'http://cern.ch/foo/plots', where 'foo' is the username
~~~


# Running on condor

The reco sequence takes about 1-2 hours / 100 events on batch. We have prepared condor scripts to facilitate this on lxbatch. 
~~~
cd $CMSSW_BASE/src/Validation/RecoParticleFlow
mkdir -p tmp/QCD/log
cd tmp/QCD

condor_submit $CMSSW_BASE/src/Validation/RecoParticleFlow/test/QCD.jdl

#do the DQM, postprocessing and plotting steps
cd ${CMSSW_BASE}/src/Validation/RecoParticleFlow

make QCD_dqm QCD_post plots
~~~

# Running via crab

The reco step can also be run via Crab, using
~~~
cd crab
python multicrab.py
~~~

Take note that the CMSSW python configuration for running the RECO sequence is dumped into `crab/step3_dump.py`.
