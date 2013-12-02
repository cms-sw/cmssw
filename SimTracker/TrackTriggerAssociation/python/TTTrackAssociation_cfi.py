import FWCore.ParameterSet.Config as cms

TTTrackAssociatorFromPixelDigis = cms.EDProducer("TTTrackAssociator_PixelDigi_",
    TTTracks = cms.VInputTag( cms.InputTag("TTTracksFromPixelDigis", "NoDup"), ),
    TTClusterTruth = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
    TTStubTruth = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
)

