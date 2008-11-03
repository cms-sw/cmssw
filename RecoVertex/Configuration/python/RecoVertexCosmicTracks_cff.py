import FWCore.ParameterSet.Config as cms

# Reco Vertex
# initialize magnetic field #########################
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVerticesFromCosmicTracks_cfi import *

vertexrecoCosmics = cms.Sequence(offlinePrimaryVerticesFromCosmicTracks)

