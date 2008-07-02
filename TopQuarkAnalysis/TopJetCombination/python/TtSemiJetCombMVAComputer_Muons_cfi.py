import FWCore.ParameterSet.Config as cms

findTtSemiJetCombMVA = cms.EDFilter("TtSemiJetCombMVAComputer",
    jets = cms.InputTag("selectedLayer1Jets"),
    nJetsMax = cms.int32(4),
    leptons = cms.InputTag("selectedLayer1Muons")
)


