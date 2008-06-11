import FWCore.ParameterSet.Config as cms

caloSimHitStudy = cms.EDFilter("CaloSimHitStudy",
    HCCollection = cms.untracked.string('HcalHits'),
    ModuleLabel = cms.untracked.string('g4SimHits'),
    MaxEnergy = cms.untracked.double(50.0),
    EECollection = cms.untracked.string('EcalHitsEE'),
    EBCollection = cms.untracked.string('EcalHitsEB'),
    ESCollection = cms.untracked.string('EcalHitsES')
)


