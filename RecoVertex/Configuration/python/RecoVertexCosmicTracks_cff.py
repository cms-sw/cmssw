import FWCore.ParameterSet.Config as cms

# Reco Vertex
# initialize magnetic field #########################
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *

import RecoVertex.PrimaryVertexProducer.OfflinePrimaryVerticesFromCosmicTracks_cfi
offlinePrimaryVertices = RecoVertex.PrimaryVertexProducer.OfflinePrimaryVerticesFromCosmicTracks_cfi.offlinePrimaryVerticesFromCosmicTracks.clone()

vertexrecoCosmicsTask = cms.Task(offlinePrimaryVertices)

# foo bar baz
# cZa94sjzYnWCR
# pkaF54aoxJBxK
