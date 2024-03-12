import FWCore.ParameterSet.Config as cms

TTTrackAssociatorFromPixelDigis = cms.EDProducer("TTTrackAssociator_Phase2TrackerDigi_",
    TTTracks = cms.VInputTag( cms.InputTag("TTTracksFromPhase2TrackerDigis", "Level1TTTracks"),
                              #cms.InputTag("TTTracksFromPixelDigisAM", "AML1Tracks"),
    ),
    TTClusterTruth = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
    TTStubTruth = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
    TTTrackAllowOneFalse2SStub = cms.bool(True),
)

# foo bar baz
# g0H4T0js1AtI0
# A2Va2e0sO51GS
