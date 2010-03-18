#! /bin/csh

setenv SCRAM_ARCH slc5_ia32_gcc434

cd /afs/cern.ch/user/g/giamman/scratch0/tksimu/CMSSW_3_5_4/src/SimTracker/SiStripDigitizer/test
eval `scramv1 runtime -csh`

nsrm /castor/cern.ch/user/g/giamman/test/step2_RAW2DIGI_L1Reco_RECO_ALCA_VALIDATION.root

cmsRun step2_RAW2DIGI_L1Reco_RECO_ALCA_VALIDATION.py
cmsRun step3_HARVESTING_MC.py



