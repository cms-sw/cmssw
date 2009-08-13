#!/bin/sh
cd /afs/cern.ch/user/p/pjanot/scratch0/CMSSW_3_1_2/src
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
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_60.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_61.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_62.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_63.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_64.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_65.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_66.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_67.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_68.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_69.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_70.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_71.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_72.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_73.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_74.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_75.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_76.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_77.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_78.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_79.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_80.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_81.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_82.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_83.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_84.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_85.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_86.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_87.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_88.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_89.root'
     ),
     noEventSort = cms.untracked.bool(True),
     duplicateCheckMode = cms.untracked.string('noDuplicateCheck')

)

process.reco = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('reco_QCDForPF_Full_003.root'),
    outputCommands = cms.untracked.vstring(
        'keep *'
    )
)

process.outpath = cms.EndPath(process.reco)

EOF
cmsRun TEST_cfg.py

rfcp reco_QCDForPF_Full_003.root /castor/cern.ch/user/p/pjanot/CMSSW312/reco_QCDForPF_Full_003.root
#rm reco_QCDForPF_Full_003.root

