import FWCore.ParameterSet.Config as cms

#
# module to make the mvaComputer
#
findTtSemiLepSignalSelMVA = cms.EDProducer("TtSemiLepSignalSelMVAComputer",
    jets     = cms.InputTag("selectedLayer1Jets"),
    leptons  = cms.InputTag("selectedLayer1Muons"),
    METs     = cms.InputTag("selectedLayer1METs"),

    nJetsMax = cms.int32(5)
)
