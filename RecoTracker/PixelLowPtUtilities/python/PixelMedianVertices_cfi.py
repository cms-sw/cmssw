import FWCore.ParameterSet.Config as cms

pixelMedianVertices = cms.EDProducer( "PixelVertexProducerMedian",
     TrackCollection = cms.string( "pixelTracks" ),
     PtMin = cms.double( 0.5 )
)
