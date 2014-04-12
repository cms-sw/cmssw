
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
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_0.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_1.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_2.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_3.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_4.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_5.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_6.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_7.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_8.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_9.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_10.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_11.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_12.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_13.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_14.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_15.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_16.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_17.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_18.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_19.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_20.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_21.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_22.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_23.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_24.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_25.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_26.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_27.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_28.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_29.root'
     ),
     noEventSort = cms.untracked.bool(True),
     duplicateCheckMode = cms.untracked.string('noDuplicateCheck')

)

process.aod = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/tmp/pjanot/aod_QCDForPF_Full_001.root'),
    outputCommands = cms.untracked.vstring(
        'keep *'
    )
)

process.outpath = cms.EndPath(process.aod)

EOF
cmsRun TEST_cfg.py

rfcp /tmp/pjanot/aod_QCDForPF_Full_001.root /castor/cern.ch/user/p/pjanot/CMSSW3100pre6/aod_QCDForPF_Full_001.root

