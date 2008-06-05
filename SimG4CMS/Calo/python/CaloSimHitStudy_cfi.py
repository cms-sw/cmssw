import FWCore.ParameterSet.Config as cms

caloSimHitStudy = cms.EDFilter("CaloSimHitStudy",
    ModuleLabel  = cms.untracked.string('g4SimHits'),
    EBCollection = cms.untracked.string('EcalHitsEB'),
    EECollection = cms.untracked.string('EcalHitsEE'),
    ESCollection = cms.untracked.string('EcalHitsES'),
    HCCollection = cms.untracked.string('HcalHits'),
    MaxEnergy    = cms.untracked.double(50.0)
)
