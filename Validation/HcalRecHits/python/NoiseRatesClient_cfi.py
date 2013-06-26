import FWCore.ParameterSet.Config as cms

noiseratesClient = cms.EDAnalyzer("NoiseRatesClient", 
#     outputFile = cms.untracked.string('NoiseRatesHarvestingME.root'),
     outputFile = cms.untracked.string(''),
     DQMDirName = cms.string("/") # root directory
)
