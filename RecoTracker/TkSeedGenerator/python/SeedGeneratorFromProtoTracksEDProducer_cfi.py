import FWCore.ParameterSet.Config as cms

seedsFromProtoTracks = cms.EDProducer("SeedGeneratorFromProtoTracksEDProducer",
  InputCollection = cms.InputTag("pixelTracks"),
  useProtoTrackKinematics = cms.bool(False),
  TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets') 
)


