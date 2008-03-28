import FWCore.ParameterSet.Config as cms

# Reco Vertex
# initialize magnetic field #########################
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
import copy
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVerticesFromCTFTracks_cfi import *
offlinePrimaryVerticesFromCTFTracksAVF = copy.deepcopy(offlinePrimaryVerticesFromCTFTracks)
import copy
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVerticesFromCTFTracks_cfi import *
offlinePrimaryVerticesFromCTFTracksKVF = copy.deepcopy(offlinePrimaryVerticesFromCTFTracks)
import copy
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVerticesFromCTFTracks_cfi import *
offlinePrimaryVerticesFromCTFTracksTKF = copy.deepcopy(offlinePrimaryVerticesFromCTFTracks)
#include "Validation/RecoVertex/data/OffLinePVFromRSTracks.cfi"
vertexreco = cms.Sequence(offlinePrimaryVerticesFromCTFTracksAVF*offlinePrimaryVerticesFromCTFTracksKVF)
offlinePrimaryVerticesFromCTFTracksAVF.algorithm = 'AdaptiveVertexFitter'
offlinePrimaryVerticesFromCTFTracksKVF.algorithm = 'KalmanVertexFitter'
offlinePrimaryVerticesFromCTFTracksTKF.algorithm = 'TrimmedKalmanFinder'

