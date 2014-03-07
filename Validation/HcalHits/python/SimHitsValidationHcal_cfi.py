import FWCore.ParameterSet.Config as cms

simHitsValidationHcal = cms.EDAnalyzer("SimHitsValidationHcal",
    moduleLabel = cms.untracked.string('g4SimHits'),
    TestNumber  = cms.untracked.bool(False),
    Verbose = cms.untracked.bool(False),
    HitCollection = cms.untracked.string('HcalHits')
)



