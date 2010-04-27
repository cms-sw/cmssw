#! /bin/csh

setenv SCRAM_ARCH slc5_ia32_gcc434

cd /afs/cern.ch/user/g/giamman/scratch0/tkvalid/CMSSW_3_7_0_pre1/src/Validation/TrackerConfiguration/test
eval `scramv1 runtime -csh`

### to do list:
# cmsDriver.py step2 -s RAW2DIGI,L1Reco,RECO,VALIDATION --relval 25000,100 --datatier GEN-SIM-RECO --eventcontent RECOSIM --geometry DB --conditions auto:mc --filein /store/relval/CMSSW_3_6_0_pre2/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24_GeomDB-v1/0000/74D1A746-DC27-DF11-85C2-0030487CD77E.root
# nota bene: input file must be from the sample on dbs
# add all input files from DBS
# edit number of events
# output in /tmp/giamman/
# cmsDriver.py step3 -s HARVESTING:validationHarvesting --conditions FrontierConditions_GlobalTag,MC_3XY_V24::All --filein file:step2_RAW2DIGI_L1Reco_RECO_VALIDATION.root
# nota bene: global tag has to be updated
# edit number of events
# input from 'file:/tmp/giamman/'

Download the already harvested files from here: 

nsrm /castor/cern.ch/user/g/giamman/test/step2_RAW2DIGI_L1Reco_RECO_VALIDATION.root

cmsRun step2_RAW2DIGI_L1Reco_RECO_VALIDATION.py
rfcp /tmp/giamman/step2_RAW2DIGI_L1Reco_RECO_VALIDATION.root /castor/cern.ch/user/g/giamman/test/
cmsRun step3_HARVESTING_MC.py



