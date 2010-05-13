#! /bin/csh

setenv SCRAM_ARCH slc5_ia32_gcc434

cd /afs/cern.ch/user/g/giamman/scratch0/tkvalid/CMSSW_3_7_0_pre4/src/Validation/TrackerConfiguration/test
eval `scramv1 runtime -csh`

# cmsDriver.py step3 -s HARVESTING:validationHarvesting+dqmHarvesting --harvesting AtRunEnd --conditions auto:mc --filein /store/relval/CMSSW_3_7_0_pre4/RelValSingleMuPt10/GEN-SIM-RECO/MC_37Y_V3-v1/0020/B21CAF9D-4B5D-DF11-A968-001BFCDBD1BA.root --mc
# edit number of events

cmsRun step3_HARVESTING.py



