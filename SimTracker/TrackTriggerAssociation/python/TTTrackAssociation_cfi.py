import FWCore.ParameterSet.Config as cms

TTTrackAssociatorFromPhase2TrackerDigis = cms.EDProducer("TTTrackAssociator_Phase2TrackerDigi_",
    TTTracks = cms.VInputTag( cms.InputTag("TTTracksFromPhase2TrackerDigis", "Level1TTTracks"),
                              #cms.InputTag("TTTracksFromPhase2TrackerDigisAM", "AML1Tracks"),
    ),
    TTClusterTruth = cms.InputTag("TTClusterAssociatorFromPhase2TrackerDigis", "ClusterAccepted"),
    TTStubTruth = cms.InputTag("TTStubAssociatorFromPhase2TrackerDigis", "StubAccepted"),
)

