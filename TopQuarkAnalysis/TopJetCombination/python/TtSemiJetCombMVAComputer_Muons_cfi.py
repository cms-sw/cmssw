import FWCore.ParameterSet.Config as cms

#
# module to make the mvaComputer
#
findTtSemiJetCombMVA = cms.EDProducer("TtSemiJetCombMVAComputer",
    jets     = cms.InputTag("selectedLayer1Jets"),
    leptons  = cms.InputTag("selectedLayer1Muons"),
    nJetsMax = cms.int32(4)
)
