import FWCore.ParameterSet.Config as cms

ttSemiHypothesisMVADisc = cms.EDFilter("TtSemiHypothesisMVADisc",
    jets = cms.InputTag("selectedLayer1Jets"),
    mets = cms.InputTag("selectedLayer1METs"),
    match = cms.InputTag("findTtSemiJetCombMVA"),
    leps = cms.InputTag("selectedLayer1Muons")
)


