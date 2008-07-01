import FWCore.ParameterSet.Config as cms

genEvt_SingleTop = cms.EDProducer("StGenEventReco",
    src  = cms.InputTag("decaySubset"),
    init = cms.InputTag("initSubset")
)
