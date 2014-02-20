import FWCore.ParameterSet.Config as cms

TTTrackAssociatorFromPixelDigis = cms.EDProducer("TTTrackAssociator_PixelDigi_",
    TTTracks = cms.VInputTag( cms.InputTag("TTTracksFromPixelDigis", "Level1TTTracks"),
                              #cms.InputTag("TTTracksFromPixelDigisAM", "AML1Tracks"),
    ),
    TTClusterTruth = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
    TTStubTruth = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
)

