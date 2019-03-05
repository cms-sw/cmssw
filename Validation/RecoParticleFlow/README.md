
# Quickstart

Run locally on lxplus
~~~
#set up the work area
export SCRAM_ARCH=slc6_amd64_gcc700
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

#Do final HTML plots
make plots
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
