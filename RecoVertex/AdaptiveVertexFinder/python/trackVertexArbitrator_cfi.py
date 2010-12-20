import FWCore.ParameterSet.Config as cms

trackVertexArbitrator = cms.EDProducer("TrackVertexArbitrator",
       beamSpot = cms.InputTag("offlineBeamSpot"),
       primaryVertices = cms.InputTag("offlinePrimaryVertices"),
       tracks = cms.InputTag("generalTracks"),
       secondaryVertices = cms.InputTag("vertexMerger"),
       dLenFraction = cms.double(0.333),
       dRCut = cms.double(0.4),
       distCut = cms.double(0.01),
       sigCut = cms.double(5)
)


