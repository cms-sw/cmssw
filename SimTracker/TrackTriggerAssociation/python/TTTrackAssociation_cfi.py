import FWCore.ParameterSet.Config as cms

TTTrackAssociatorFromPixelDigis = cms.EDProducer("TTTrackAssociator_Phase2TrackerDigi_",
    TTTracks = cms.VInputTag( cms.InputTag("TTTracksFromPhase2TrackerDigis", "Level1TTTracks"),
                              #cms.InputTag("TTTracksFromPixelDigisAM", "AML1Tracks"),
    ),
    TTClusterTruth = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
    TTStubTruth = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
                                                 # The following cfg param has been removed, as it was unnecessary.
                                                 # You can specify if an incorrect 2S or PS stub are allowed
                                                 # via function TTTrackAssociationMap::isGenuine().
    #TTTrackAllowOneFalse2SStub = cms.bool(True),
)

