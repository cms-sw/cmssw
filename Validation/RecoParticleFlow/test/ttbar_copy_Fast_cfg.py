import FWCore.ParameterSet.Config as cms
 

process = cms.Process("COPY")
 

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
 
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Fast_1.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Fast_2.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Fast_3.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Fast_4.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Fast_5.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Fast_6.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Fast_7.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Fast_8.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Fast_9.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Fast_10.root'
     )
)

process.display = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('aod_ttbar_Fast.root'),
    outputCommands = cms.untracked.vstring(
        'keep *'
    )
)

process.outpath = cms.EndPath(process.display)

