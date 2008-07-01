import FWCore.ParameterSet.Config as cms

genEvt = cms.EDProducer("TtGenEventReco",
    src  = cms.InputTag("decaySubset"),
    init = cms.InputTag("initSubset")
)


