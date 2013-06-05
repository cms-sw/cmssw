import FWCore.ParameterSet.Config as cms

# Magnetic Field, Geometry, TransientTracks
from Configuration.StandardSequences.MagneticField_cff import *

# Track Associators
from SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi import *
from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *

# Track history parameters
trackHistory = cms.PSet(
    bestMatchByMaxValue = cms.untracked.bool(True),
    trackingTruth = cms.untracked.InputTag("mix","MergedTrackTruth"),
    trackAssociator = cms.untracked.string('TrackAssociatorByHits'),
    trackProducer = cms.untracked.InputTag("generalTracks"),
    enableRecoToSim = cms.untracked.bool(True),
    enableSimToReco = cms.untracked.bool(False)
)


