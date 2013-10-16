import FWCore.ParameterSet.Config as cms

TTTrackAssociatorFromPixelDigis = cms.EDProducer("TTTrackAssociator_PixelDigi_",
    TTTracks = cms.InputTag("TTTracksFromPixelDigis", "NoDup"),
    TTSeeds = cms.InputTag("TTTracksFromPixelDigis", "Seeds"),
    TTClusterTruth = cms.InputTag("TTClusterAssociatorFromPixelDigis"),
    TTStubTruth = cms.InputTag("TTStubAssociatorFromPixelDigis"),
)

