import FWCore.ParameterSet.Config as cms

calotowersAnalyzer = DQMStep1Module('CaloTowersValidation',
    outputFile               = cms.untracked.string(''),
    CaloTowerCollectionLabel = cms.untracked.InputTag('towerMaker'),
    hcalselector             = cms.untracked.string('all'),
    mc                       = cms.untracked.string('yes')
)



