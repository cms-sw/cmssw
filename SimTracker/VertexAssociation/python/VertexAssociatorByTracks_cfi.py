import FWCore.ParameterSet.Config as cms

VertexAssociatorByTracksESProducer = cms.ESProducer("VertexAssociatorByTracksESProducer",
    MinTrackFraction = cms.double(0.5)
)


