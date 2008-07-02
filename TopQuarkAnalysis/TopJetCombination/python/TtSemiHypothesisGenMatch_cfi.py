import FWCore.ParameterSet.Config as cms

ttSemiHypothesisGenMatch = cms.EDFilter("TtSemiHypothesisGenMatch",
    jets = cms.InputTag("selectedLayer1Jets"),
    mets = cms.InputTag("selectedLayer1METs"),
    match = cms.InputTag("ttSemiJetPartonMatch"),
    leps = cms.InputTag("selectedLayer1Muons")
)


