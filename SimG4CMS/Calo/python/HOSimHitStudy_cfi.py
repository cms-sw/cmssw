import FWCore.ParameterSet.Config as cms

hoSimHitStudy = cms.EDFilter("HOSimHitStudy",
    ModuleLabel = cms.untracked.string('g4SimHits'),
    EBCollection = cms.untracked.string('EcalHitsEB'),
    HCCollection = cms.untracked.string('HcalHits'),
    MaxEnergy = cms.untracked.double(50.0),
    ScaleEB   = cms.untracked.double(1.02),
    ScaleHB   = cms.untracked.double(104.4),
    ScaleHO   = cms.untracked.double(2.33)
)


