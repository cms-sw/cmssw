import FWCore.ParameterSet.Config as cms

ttSemiHypothesisMaxSumPtWMass = cms.EDFilter("TtSemiHypothesisMaxSumPtWMass",
    jets = cms.InputTag("selectedLayer1Jets"),
    mets = cms.InputTag("selectedLayer1METs"),
    maxNJets = cms.uint32(4),
    match = cms.InputTag("ttSemiJetPartonMatch"),
    leps = cms.InputTag("selectedLayer1Muons")
)


