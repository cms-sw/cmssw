import FWCore.ParameterSet.Config as cms

TTTrackAssociatorFromPixelDigis = cms.EDProducer("TTTrackAssociator_PixelDigi_",
    TTTracks = cms.VInputTag( #cms.InputTag("TTTracksFromPixelDigisTracklet", "TrackletBasedL1Tracks"),
                              cms.InputTag("TTTracksFromPixelDigisAM", "AML1Tracks"),
    ),
    TTClusterTruth = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
    TTStubTruth = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
)

