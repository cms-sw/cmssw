import FWCore.ParameterSet.Config as cms

hcalrechitsClient = cms.EDProducer("HcalRecHitsClient", 
#     outputFile = cms.untracked.string('HcalRecHitsHarvestingME.root'),
     outputFile = cms.untracked.string(''),
     DQMDirName = cms.string("/") # root directory
)
