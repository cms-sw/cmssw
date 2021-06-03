import FWCore.ParameterSet.Config as cms

from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import *
from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import *

from Configuration.Eras.Modifier_fastSim_cff import fastSim

prunedTpClusterProducer = tpClusterProducer.clone(
    trackingParticleSrc = cms.InputTag("prunedTrackingParticles"),
    pixelSimLinkSrc = cms.InputTag("prunedDigiSimLinks", "siPixel"),
    stripSimLinkSrc = cms.InputTag("prunedDigiSimLinks", "siStrip"),
    throwOnMissingCollections = cms.bool(False)
)

quickPrunedTrackAssociatorByHits = quickTrackAssociatorByHits.clone(
    cluster2TPSrc = "prunedTpClusterProducer"
)

prunedTrackMCMatch = cms.EDProducer("MCTrackMatcher",
    trackingParticles = cms.InputTag("prunedTrackingParticles"),
    tracks = cms.InputTag("generalTracks"),
    genParticles = cms.InputTag("genParticles"),
    associator = cms.string('quickPrunedTrackAssociatorByHits'),
    throwOnMissingTPCollection = cms.bool(False)
)

trackPrunedMCMatchTask = cms.Task(prunedTpClusterProducer,quickPrunedTrackAssociatorByHits,prunedTrackMCMatch)



from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(quickTrackAssociatorByHits, quickPrunedTrackAssociatorByHits.clone(
    useClusterTPAssociation = cms.bool(False),
    associateStrip = cms.bool(False),
    associatePixel = cms.bool(False),
))

fastSim.toModify(trackPrunedMCMatchTask, lambda x: x.remove(prunedTpClusterProducer))
