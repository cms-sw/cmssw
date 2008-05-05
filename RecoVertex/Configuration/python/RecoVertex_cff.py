import FWCore.ParameterSet.Config as cms

# Reco Vertex
# initialize magnetic field #########################
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import *
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVerticesWithBS_cfi import *
vertexreco = cms.Sequence(offlinePrimaryVertices*offlinePrimaryVerticesWithBS)

