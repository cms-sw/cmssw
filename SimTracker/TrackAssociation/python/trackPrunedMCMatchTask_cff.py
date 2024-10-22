import FWCore.ParameterSet.Config as cms

from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import *
from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import *

from Configuration.Eras.Modifier_fastSim_cff import fastSim

prunedTpClusterProducer = tpClusterProducer.clone(
    trackingParticleSrc = cms.InputTag("prunedTrackingParticles"),
    pixelSimLinkSrc = cms.InputTag("prunedDigiSimLinks", "siPixel"),
    stripSimLinkSrc = cms.InputTag("prunedDigiSimLinks", "siStrip"),
    throwOnMissingCollections = cms.bool(True)
)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
_phase2_prunedTpClusterProducer = tpClusterProducer.clone(
    trackingParticleSrc = cms.InputTag("prunedTrackingParticles"),
    pixelSimLinkSrc = cms.InputTag("prunedDigiSimLinks", "siPixel"),
    phase2OTSimLinkSrc = cms.InputTag("prunedDigiSimLinks", "siphase2OT"),
    throwOnMissingCollections = cms.bool(True)
)
phase2_tracker.toReplaceWith( 
    prunedTpClusterProducer,
    _phase2_prunedTpClusterProducer
)

quickPrunedTrackAssociatorByHits = quickTrackAssociatorByHits.clone(
    cluster2TPSrc = "prunedTpClusterProducer"
)

prunedTrackMCMatch = cms.EDProducer("MCTrackMatcher",
    trackingParticles = cms.InputTag("prunedTrackingParticles"),
    tracks = cms.InputTag("generalTracks"),
    genParticles = cms.InputTag("genParticles"),
    associator = cms.string('quickPrunedTrackAssociatorByHits'),
    throwOnMissingTPCollection = cms.bool(True)
)

trackPrunedMCMatchTask = cms.Task(prunedTpClusterProducer,quickPrunedTrackAssociatorByHits,prunedTrackMCMatch)



from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(quickTrackAssociatorByHits, quickPrunedTrackAssociatorByHits.clone(
    useClusterTPAssociation = cms.bool(False),
    associateStrip = cms.bool(False),
    associatePixel = cms.bool(False),
))

fastSim.toModify(trackPrunedMCMatchTask, lambda x: x.remove(prunedTpClusterProducer))
