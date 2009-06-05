import FWCore.ParameterSet.Config as cms

#
# module to make the mvaComputer
#
findTtSemiLepSignalSelMVA = cms.EDProducer("TtSemiLepSignalSelMVAComputer",
    jets     = cms.InputTag("selectedLayer1Jets"),
    muons  = cms.InputTag("selectedLayer1Muons"),
    METs     = cms.InputTag("selectedLayer1METs"),
    electrons  = cms.InputTag("selectedLayer1Electrons")
)
