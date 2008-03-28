import FWCore.ParameterSet.Config as cms

seedsFromProtoTracks = cms.EDProducer("SeedGeneratorFromProtoTracksEDProducer",
    TTRHBuilder = cms.string('WithTrackAngle'),
    InputCollection = cms.InputTag("pixelTracks")
)


