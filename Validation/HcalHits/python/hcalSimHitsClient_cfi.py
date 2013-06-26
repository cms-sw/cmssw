import FWCore.ParameterSet.Config as cms

hcalsimhitsClient = cms.EDAnalyzer("HcalSimHitsClient", 
     outputFile = cms.untracked.string(''),
     DQMDirName = cms.string("/") # root directory
)
