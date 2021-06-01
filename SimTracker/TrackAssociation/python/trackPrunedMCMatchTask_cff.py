import FWCore.ParameterSet.Config as cms

from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import *
from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import *

prunedTpClusterProducer = tpClusterProducer.clone(
    trackingParticleSrc = cms.InputTag("prunedTrackingParticles"),
    pixelSimLinkSrc = cms.InputTag("prunedDigiSimLinks", "siPixel"),
    stripSimLinkSrc = cms.InputTag("prunedDigiSimLinks", "siStrip")
)

quickPrunedTrackAssociatorByHits = quickTrackAssociatorByHits.clone(
    cluster2TPSrc = "prunedTpClusterProducer"
)

prunedTrackMCMatch = cms.EDProducer("MCTrackMatcher",
    trackingParticles = cms.InputTag("prunedTrackingParticles"),
    tracks = cms.InputTag("generalTracks"),
    genParticles = cms.InputTag("genParticles"),
    associator = cms.string('quickPrunedTrackAssociatorByHits')
)

trackPrunedMCMatchTask = cms.Task(prunedTpClusterProducer,quickPrunedTrackAssociatorByHits,prunedTrackMCMatch)
