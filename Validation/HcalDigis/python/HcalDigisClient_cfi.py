import FWCore.ParameterSet.Config as cms

hcaldigisClient = cms.EDProducer("HcalDigisClient",
     outputFile = cms.untracked.string(''),
     DQMDirName = cms.string("/") # root directory
)
