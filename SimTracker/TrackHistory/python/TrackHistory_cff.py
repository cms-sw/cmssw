import FWCore.ParameterSet.Config as cms

# Track Associators
from SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi import *
from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
# Track history parameters
trackHistory = cms.PSet(
    associationModule = cms.string('TrackAssociatorByHits'),
    trackingParticleProduct = cms.string(''),
    trackingParticleModule = cms.string('trackingtruthprod'),
    bestMatchByMaxValue = cms.bool(True),
    recoTrackModule = cms.string('ctfWithMaterialTracks')
)

