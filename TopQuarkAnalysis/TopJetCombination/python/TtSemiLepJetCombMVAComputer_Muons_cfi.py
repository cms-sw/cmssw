import FWCore.ParameterSet.Config as cms

#
# module to make the mvaComputer
#
findTtSemiLepJetCombMVA = cms.EDProducer("TtSemiLepJetCombMVAComputer",
    jets     = cms.InputTag("selectedLayer1Jets"),
    leptons  = cms.InputTag("selectedLayer1Muons"),
    maxNJets = cms.int32(4)
)
