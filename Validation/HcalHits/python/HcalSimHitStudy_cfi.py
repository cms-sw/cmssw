import FWCore.ParameterSet.Config as cms

hcalSimHitStudy = cms.EDFilter("HcalSimHitStudy",
    ModuleLabel = cms.untracked.string('g4SimHits'),
    outputFile = cms.untracked.string(''),
    Verbose = cms.untracked.bool(False),
    HitCollection = cms.untracked.string('HcalHits')
)



