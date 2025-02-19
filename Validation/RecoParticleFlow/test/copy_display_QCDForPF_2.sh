#!/bin/sh
cd /afs/cern.ch/user/p/pjanot/scratch0/CMSSW_3_10_0_pre6/src
eval `scramv1 runtime -sh`
cd -
#commande pour decoder le .cfg
cat > TEST_cfg.py << "EOF"
import FWCore.ParameterSet.Config as cms
 

process = cms.Process("COPY")
 

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
 
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_30.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_31.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_32.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_33.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_34.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_35.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_36.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_37.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_38.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_39.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_40.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_41.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_42.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_43.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_44.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_45.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_46.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_47.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_48.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_49.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_50.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_51.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_52.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_53.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_54.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_55.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_56.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_57.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_58.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_59.root'
     ),
     noEventSort = cms.untracked.bool(True),
     duplicateCheckMode = cms.untracked.string('noDuplicateCheck')

)

process.display = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/tmp/pjanot/display_QCDForPF_Full_002.root'),
    outputCommands = cms.untracked.vstring(
        'keep *'
    )
)

process.outpath = cms.EndPath(process.display)

EOF
cmsRun TEST_cfg.py

rfcp /tmp/pjanot/display_QCDForPF_Full_002.root /castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_002.root


