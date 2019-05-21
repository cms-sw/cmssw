
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
#Necessary if you are 
make QCD_reco

#DQM step, a few minutes
make QCD_dqm

# Repeat for QCDPU & NuGunPU (make QCDPU_reco etc.) or edit the 'make plots' part of
# Makefile for successfully running 'make plots'

#Do final HTML plots
make plots
# or if you have tmp/QCD_ref, tmp/QCDPU_ref, tmp/NuGunPU_ref (i.e. reference results) etc under _tmp area
make pltos_with_ref


# The 'plots' directory can be viewed from a web browser once it is moved to e.g. /afs/cern.ch/user/f/foo/www/.
# In this case the URL for the directory is 'http://cern.ch/foo/plots', where 'foo' is the username
~~~


# Running via crab

The reco step can also be run via Crab, using
~~~
cd crab
python multicrab.py
~~~

Take note that the CMSSW python configuration for running the RECO sequence is dumped into `crab/step3_dump.py`.


# Running DQM steps from existing MINIAOD samples

~~~

# For example (for 2017):
CONDITIONS=auto:phase1_2017_realistic
ERA=Run2_2017
#Running with 2 threads allows to use more memory on grid
NTHREADS=2
TMPDIR=tmp

cd $CMSSW_BASE/src/Validation/RecoParticleFlow
cd tmp/QCD

# make a text file for input files
dasgoclient --query="file dataset=/RelValNuGun/CMSSW_10_5_0_pre1-PU25ns_103X_upgrade2018_realistic_v8-v1/MINIAODSIM" > step3_filelist.txt
cat step3_filelist.txt

cmsDriver.py step5 --conditions $CONDITIONS -s DQM:@pfDQM --datatier DQMIO --nThreads $NTHREADS --era $ERA --eventcontent DQM --filein filelist:step3_filelist.txt --fileout file:step5.root -n -1 2>&1 | tee step5.log
cmsDriver.py step6 --conditions $CONDITIONS -s HARVESTING:@pfDQM --era $ERA --filetype DQM --filein file:step5.root --fileout file:step6.root
