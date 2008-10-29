import FWCore.ParameterSet.Config as cms

hfPMTHitAnalyzer = cms.EDFilter("HFPMTHitAnalyzer",
    moduleLabel = cms.untracked.string('g4SimHits'),
    HitCollection = cms.untracked.string('HcalHits')
)


