import FWCore.ParameterSet.Config as cms

calotowersClient = cms.EDAnalyzer("CaloTowersClient",
#     outputFile = cms.untracked.string('CaloTowersHarvestingME.root'),
     outputFile = cms.untracked.string(''),
     DQMDirName = cms.string("/") # root directory
)
