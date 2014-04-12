import FWCore.ParameterSet.Config as cms

# Reco Vertex
# initialize magnetic field #########################
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
import RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi
offlinePrimaryVerticesFromCTFTracksAVF = RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi.offlinePrimaryVertices.clone()
import RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi
offlinePrimaryVerticesFromCTFTracksKVF = RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi.offlinePrimaryVertices.clone()

vertexreco = cms.Sequence(offlinePrimaryVerticesFromCTFTracksAVF*offlinePrimaryVerticesFromCTFTracksKVF)
offlinePrimaryVerticesFromCTFTracksAVF.vertexCollections[0].algorithm = 'AdaptiveVertexFitter'
offlinePrimaryVerticesFromCTFTracksKVF.vertexCollections[0].algorithm = 'KalmanVertexFitter'

