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
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_90.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_91.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_92.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_93.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_94.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_95.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_96.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_97.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_98.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_99.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_100.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_101.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_102.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_103.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_104.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_105.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_106.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_107.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_108.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_109.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_110.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_111.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_112.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_113.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_114.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_115.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_116.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_117.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_118.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_119.root'
     ),
     noEventSort = cms.untracked.bool(True),
     duplicateCheckMode = cms.untracked.string('noDuplicateCheck')

)

process.aod = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/tmp/pjanot/aod_QCDForPF_Full_004.root'),
    outputCommands = cms.untracked.vstring(
        'keep *'
    )
)

process.outpath = cms.EndPath(process.aod)

EOF
cmsRun TEST_cfg.py

rfcp /tmp/pjanot/aod_QCDForPF_Full_004.root /castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_004.root

