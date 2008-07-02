import FWCore.ParameterSet.Config as cms

ttSemiJetPartonMatch = cms.EDFilter("TtSemiEvtJetPartonMatch",
    maxDist = cms.double(0.3),
    useMaxDist = cms.bool(False),
    algorithm = cms.int32(0),
    nJets = cms.int32(-1),
    useDeltaR = cms.bool(True),
    jets = cms.InputTag("selectedLayer1Jets")
)


