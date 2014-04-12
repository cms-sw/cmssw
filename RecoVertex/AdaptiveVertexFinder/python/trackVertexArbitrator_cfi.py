import FWCore.ParameterSet.Config as cms

trackVertexArbitrator = cms.EDProducer("TrackVertexArbitrator",
       beamSpot = cms.InputTag("offlineBeamSpot"),
       primaryVertices = cms.InputTag("offlinePrimaryVertices"),
       tracks = cms.InputTag("generalTracks"),
       secondaryVertices = cms.InputTag("vertexMerger"),
       dLenFraction = cms.double(0.333),
       dRCut = cms.double(0.4),
       distCut = cms.double(0.04),
       sigCut = cms.double(5),
       fitterSigmacut =  cms.double(3),
       fitterTini = cms.double(256),
       fitterRatio = cms.double(0.25),
       trackMinLayers = cms.int32(4),
       trackMinPt = cms.double(0.4),
       trackMinPixels = cms.int32(1)

)


