import FWCore.ParameterSet.Config as cms

# Magnetic Field, Geometry, TransientTracks
from Configuration.StandardSequences.MagneticField_cff import *

# Track Associators
from SimTracker.TrackAssociatorProducers.trackAssociatorByChi2_cfi import *
from SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi import *
from SimTracker.VertexAssociation.VertexAssociatorByTracks_cfi import *

# Track history parameters
vertexHistory = cms.PSet(
    bestMatchByMaxValue = cms.untracked.bool(True),
    trackingTruth = cms.untracked.InputTag('mix','MergedTrackTruth'),
    trackAssociator = cms.untracked.InputTag('trackAssociatorByHits'),
    trackProducer = cms.untracked.InputTag('generalTracks'),
    vertexAssociator = cms.untracked.string('VertexAssociatorByTracks'),
    vertexProducer = cms.untracked.InputTag('offlinePrimaryVertices'),
    enableRecoToSim = cms.untracked.bool(True),
    enableSimToReco = cms.untracked.bool(False)
)


