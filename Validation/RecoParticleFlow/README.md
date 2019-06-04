
# Quickstart

Run locally on lxplus
~~~
#set up the work area
# for lxplus with SLC7 (default since April 2019)
export SCRAM_ARCH=slc7_amd64_gcc700
# for SLC6 use 'slc6_amd64_gcc700' instead above
cmsrel CMSSW_10_6_0
cd CMSSW_10_6_0
cmsenv

#get the code
git cms-checkout-topic jpata:pfvalidation-10_6_X-master
# 'git cms-rebase-topic jpata:pfvalidation-10_6_X-master' might be better when using 10_6_0_patchX or higher version
scram b -j4
cd $CMSSW_BASE/src/Validation/RecoParticleFlow

# Activate reading files from remote locations (needed at lxplus at least)
voms-proxy-init -voms cms

#RECO step, about 30 minutes
#Necessary if you need to re-reco events to test introduced changes to PF reco.
#Note that the default era & condition is now set to 2018. Change CONDITIONS and ERA in test/run_relval.sh when trying other era, before trying the above commands.
make QCD_reco

#DQM step, a few minutes
make QCD_dqm

# Repeat for QCDPU & NuGunPU (make QCDPU_reco, make QCDPU_dqm etc.) or use crab for reco and run dqm steps as indicated below.
# Or, edit the 'make plots' part of Makefile for successfully running 'make plots'

#Do final HTML plots (this just plot two identical results in tmp/{QCD,QCDPU,NuGunPU} twice)
make plots
# 
# or if you have reference DQM results in tmp/QCD_ref, tmp/QCDPU_ref, tmp/NuGunPU_ref (i.e. reference results) etc under _tmp area:
make pltos_with_ref

# The 'plots' directory can be viewed from a web browser once it is moved to e.g. /afs/cern.ch/user/f/foo/www/.
# In this case the URL for the directory is 'http://cern.ch/foo/plots', where 'foo' is the username
~~~


# Running via crab

The reco step can also be run via Crab, using
~~~
make conf
make dumpconf
cd crab
python multicrab.py
~~~

Note that the default era, condition, and samples are now set to 2018. Change CONDITIONS and ERA in test/run_relval.sh when trying other era, before trying the above commands. Also check (and if necessary, update) input samples and conf.Site.storageSite specified in $CMSSW_BASE/src/Validation/RecoParticleFlow/crab/multicrab.py.  
Take note that the CMSSW python configuration for running the RECO sequence is dumped into `crab/step3_dump.py`.


# Running DQM steps from existing MINIAOD samples

~~~

# For example (default for 2018):
#CONDITIONS=auto:phase1_2017_realistic ERA=Run2_2017 # for 2017 scenarios
CONDITIONS=auto:phase1_2018_realistic ERA=Run2_2018 # for 2018 scenarios
#CONDITIONS=auto:phase2_realistic ERA=Phase2C8_timing_layer_bar # for phase2 
#Running with 2 threads allows to use more memory on grid
NTHREADS=2 TMPDIR=tmp

cd $CMSSW_BASE/src/Validation/RecoParticleFlow
make -p tmp/QCD; cd tmp/QCD
#(or
make -p tmp/QCDPU; cd tmp/QCDPU
make -p tmp/NuGunPU; cd tmp/NuGunPU
#)

# make a text file for input files. For example:
dasgoclient --query="file dataset=/RelValQCD_FlatPt_15_3000HS_13/CMSSW_10_6_0-106X_upgrade2018_realistic_v4-v1/MINIAODSIM" > step3_filelist.txt
#(or
dasgoclient --query="file dataset=/RelValQCD_FlatPt_15_3000HS_13/CMSSW_10_6_0-PU25ns_106X_upgrade2018_realistic_v4-v1/MINIAODSIM" > step3_filelist.txt
dasgoclient --query="file dataset=/RelValNuGun/CMSSW_10_6_0-PU25ns_106X_upgrade2018_realistic_v4-v1/MINIAODSIM" > step3_filelist.txt
dasgoclient --query="file dataset=/RelValQCD_Pt-15To7000_Flat_14TeV/CMSSW_10_6_0-106X_upgrade2023_realistic_v2_2023D41noPU-v2/MINIAODSIM" > step3_filelist.txt
dasgoclient --query="file dataset=/RelValQCD_Pt-15To7000_Flat_14TeV/CMSSW_10_6_0-PU25ns_106X_upgrade2023_realistic_v2_2023D41PU200-v1/MINIAODSIM" > step3_filelist.txt
dasgoclient --query="file dataset=/RelValNuGun/CMSSW_10_6_0-PU25ns_106X_upgrade2023_realistic_v2_2023D41PU200-v1/MINIAODSIM"  > step3_filelist.txt
or using the list of files from your crab output areas.
#)
cat step3_filelist.txt

cmsDriver.py step5 --conditions $CONDITIONS -s DQM:@pfDQM --datatier DQMIO --nThreads $NTHREADS --era $ERA --eventcontent DQM --filein filelist:step3_filelist.txt --fileout file:step5.root -n -1 >& step5.log &
# after step5 is completed:
cmsDriver.py step6 --conditions $CONDITIONS -s HARVESTING:@pfDQM --era $ERA --filetype DQM --filein file:step5.root --fileout file:step6.root >& step6.log &

