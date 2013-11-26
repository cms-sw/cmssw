import FWCore.ParameterSet.Config as cms

TTStubAssociatorFromPixelDigis = cms.EDProducer("TTStubAssociator_PixelDigi_",
    TTStubs = cms.InputTag("TTStubsFromPixelDigis", "StubsPass"),
    TTClusterTruth = cms.InputTag("TTClusterAssociatorFromPixelDigis"),
)

