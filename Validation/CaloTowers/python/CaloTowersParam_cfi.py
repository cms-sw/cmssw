import FWCore.ParameterSet.Config as cms

calotowersAnalyzer = cms.EDFilter("CaloTowersValidation",
    outputFile = cms.untracked.string('CaloTowersValidationHB.root'),
    CaloTowerCollectionLabel = cms.untracked.string('towerMaker'),
    hcalselector = cms.untracked.string('HB')
)



