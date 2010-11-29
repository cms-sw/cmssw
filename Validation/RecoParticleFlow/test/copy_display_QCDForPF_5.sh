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
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_120.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_121.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_122.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_123.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_124.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_125.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_126.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_127.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_128.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_129.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_130.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_131.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_132.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_133.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_134.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_135.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_136.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_137.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_138.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_139.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_140.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_141.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_142.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_143.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_144.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_145.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_146.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_147.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_148.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_149.root'
     ),
     noEventSort = cms.untracked.bool(True),
     duplicateCheckMode = cms.untracked.string('noDuplicateCheck')

)

process.display = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/tmp/pjanot/display_QCDForPF_Full_005.root'),
    outputCommands = cms.untracked.vstring(
        'keep *'
    )
)

process.outpath = cms.EndPath(process.display)

EOF
cmsRun TEST_cfg.py

rfcp /tmp/pjanot/display_QCDForPF_Full_005.root /castor/cern.ch/user/p/pjanot/CMSSW3100pre6/display_QCDForPF_Full_005.root

