import FWCore.ParameterSet.Config as cms

hcaldigisClient = cms.EDAnalyzer("HcalDigisClient",
     outputFile = cms.untracked.string(''),
     DQMDirName = cms.string("/") # root directory
)
