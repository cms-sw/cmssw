import FWCore.ParameterSet.Config as cms

simHitsValidationHE = cms.EDAnalyzer("SimHitsValidationHE",
    moduleLabel = cms.untracked.string('g4SimHits'),
    TestNumber  = cms.untracked.bool(False),
    Verbose = cms.untracked.bool(False),
    HitCollection = cms.untracked.string('HcalHits')
)



