import FWCore.ParameterSet.Config as cms

calotowersAnalyzer = cms.EDAnalyzer("CaloTowersValidation",
    outputFile = cms.untracked.string(''),
    CaloTowerCollectionLabel = cms.untracked.string('towerMaker'),
    hcalselector = cms.untracked.string('HB')
)



