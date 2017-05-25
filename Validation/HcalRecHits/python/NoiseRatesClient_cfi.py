import FWCore.ParameterSet.Config as cms

noiseratesClient = cms.EDProducer("NoiseRatesClient", 
#     outputFile = cms.untracked.string('NoiseRatesHarvestingME.root'),
     outputFile = cms.untracked.string(''),
     DQMDirName = cms.string("/") # root directory
)
