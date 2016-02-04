import FWCore.ParameterSet.Config as cms

simHitsValidationHcal = cms.EDAnalyzer("SimHitsValidationHcal",
    ModuleLabel = cms.untracked.string('g4SimHits'),
    outputFile = cms.untracked.string(''),
    Verbose = cms.untracked.bool(False),
    HitCollection = cms.untracked.string('HcalHits')
)



