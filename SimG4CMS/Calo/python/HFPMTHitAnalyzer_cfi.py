import FWCore.ParameterSet.Config as cms

hfPMTHitAnalyzer = cms.EDFilter("HFPMTHitAnalyzer",
    SourceLabel = cms.untracked.string('generator'),
    ModuleLabel = cms.untracked.string('g4SimHits'),
    HitCollection = cms.untracked.string('HcalHits')
)


