import FWCore.ParameterSet.Config as cms

hfPMTHitAnalyzer = cms.EDAnalyzer("HFPMTHitAnalyzer",
    SourceLabel = cms.untracked.string('generatorSmeared'),
    ModuleLabel = cms.untracked.string('g4SimHits'),
    HitCollection = cms.untracked.string('HcalHits')
)


