import FWCore.ParameterSet.Config as cms

seedsFromProtoTracks = cms.EDProducer("SeedGeneratorFromProtoTracksEDProducer",
  InputCollection = cms.InputTag("pixelTracks"),
  InputVertexCollection = cms.InputTag(""),
  originHalfLength = cms.double(1E9),
  originRadius = cms.double(1E9),
  useProtoTrackKinematics = cms.bool(False),
  useEventsWithNoVertex = cms.bool(True),
  TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets') 
)


