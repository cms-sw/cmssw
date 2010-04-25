#! /bin/csh

setenv SCRAM_ARCH slc5_ia32_gcc434

cd /afs/cern.ch/user/g/giamman/scratch0/tkvalid/CMSSW_3_7_0_pre1/src/Validation/TrackerConfiguration/test
eval `scramv1 runtime -csh`

cmsRun HarvestingGlobalValidation_Tracking.py




