import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.Configuration.RecoLocalTracker_cff import *
from SimGeneral.TrackingAnalysis.simHitTPAssociation_cfi import *
from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import *
from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import *
from RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi import *
from RecoLocalTracker.Phase2TrackerRecHits.Phase2TrackerRecHits_cfi import *
from Geometry.TrackerNumberingBuilder.trackerTopology_cfi import *

from Validation.RecoTrack.trackingNtuple_cfi import *
from Validation.RecoTrack.TrackValidation_cff import *
from SimGeneral.TrackingAnalysis.trackingParticleNumberOfLayersProducer_cff import *
import Validation.RecoTrack.TrackValidation_cff as _TrackValidation_cff

_includeHits = True
#_includeHits = False

_includeSeeds = True
#_includeSeeds = False

_includeMVA = True
#_includeMVA = False

_includeTrackingParticles = True
#_includeTrackingParticles = False

from CommonTools.RecoAlgos.trackingParticleRefSelector_cfi import trackingParticleRefSelector as _trackingParticleRefSelector
trackingParticlesIntime = _trackingParticleRefSelector.clone(
    signalOnly = False,
    intimeOnly = True,
    chargedOnly = False,
    tip = 1e5,
    lip = 1e5,
    minRapidity = -10,
    maxRapidity = 10,
    ptMin = 0,
)
trackingNtuple.trackingParticles = "trackingParticlesIntime"
trackingNtuple.trackingParticlesRef = True
trackingNtuple.includeAllHits = _includeHits
trackingNtuple.includeSeeds = _includeSeeds
trackingNtuple.includeMVA = _includeMVA
trackingNtuple.includeTrackingParticles = _includeTrackingParticles

def _filterForNtuple(lst):
    ret = []
    for item in lst:
        if "PreSplitting" in item:
            continue
        if "SeedsA" in item and item.replace("SeedsA", "SeedsB") in lst:
            ret.append(item.replace("SeedsA", "Seeds"))
            continue
        if "SeedsB" in item:
            continue
        if "SeedsPair" in item and item.replace("SeedsPair", "SeedsTripl") in lst:
            ret.append(item.replace("SeedsPair", "Seeds"))
            continue
        if "SeedsTripl" in item:
            continue
        ret.append(item)
    return ret
_seedProducers = _filterForNtuple(_TrackValidation_cff._seedProducers)
_seedProducers_trackingPhase1 = _filterForNtuple(_TrackValidation_cff._seedProducers_trackingPhase1)
_seedProducers_trackingPhase1QuadProp = _filterForNtuple(_TrackValidation_cff._seedProducers_trackingPhase1QuadProp)
_seedProducers_trackingPhase2PU140  = _filterForNtuple(_TrackValidation_cff._seedProducers_trackingPhase2PU140)

(_seedSelectors, trackingNtupleSeedSelectors) = _TrackValidation_cff._addSeedToTrackProducers(_seedProducers, globals())
(_seedSelectors_trackingPhase1, _trackingNtupleSeedSelectors_trackingPhase1) = _TrackValidation_cff._addSeedToTrackProducers(_seedProducers_trackingPhase1, globals())
(_seedSelectors_trackingPhase1QuadProp, _trackingNtupleSeedSelectors_trackingPhase1QuadProp) = _TrackValidation_cff._addSeedToTrackProducers(_seedProducers_trackingPhase1QuadProp, globals())
(_seedSelectors_trackingPhase2PU140, _trackingNtupleSeedSelectors_trackingPhase2PU140) = _TrackValidation_cff._addSeedToTrackProducers(_seedProducers_trackingPhase2PU140, globals())
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
from Configuration.Eras.Modifier_trackingPhase1QuadProp_cff import trackingPhase1QuadProp
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase1.toReplaceWith(trackingNtupleSeedSelectors, _trackingNtupleSeedSelectors_trackingPhase1)
trackingPhase1QuadProp.toReplaceWith(trackingNtupleSeedSelectors, _trackingNtupleSeedSelectors_trackingPhase1QuadProp)
trackingPhase2PU140.toReplaceWith(trackingNtupleSeedSelectors, _trackingNtupleSeedSelectors_trackingPhase2PU140)

trackingNtuple.seedTracks = _seedSelectors
trackingPhase1.toModify(trackingNtuple, seedTracks = _seedSelectors_trackingPhase1)
trackingPhase1QuadProp.toModify(trackingNtuple, seedTracks = _seedSelectors_trackingPhase1)
trackingPhase2PU140.toModify(trackingNtuple, seedTracks = _seedSelectors_trackingPhase2PU140)

def _seedProdToTrackCands(name):
    return name.replace("seedTracks", "").replace("Seeds", "TrackCandidates")
trackingNtuple.trackCandidates = map(_seedProdToTrackCands, _seedProducers)
trackingPhase1.toModify(trackingNtuple, trackCandidates=map(_seedProdToTrackCands, _seedProducers_trackingPhase1))
trackingPhase1QuadProp.toModify(trackingNtuple, trackCandidates=map(_seedProdToTrackCands, _seedProducers_trackingPhase1QuadProp))
trackingPhase2PU140.toModify(trackingNtuple, trackCandidates=map(_seedProdToTrackCands, _seedProducers_trackingPhase2PU140))

trackingNtupleSequence = cms.Sequence()
# reproduce hits because they're not stored in RECO
if _includeHits:
    trackingNtupleSequence += (
        siPixelRecHits +
        siStripMatchedRecHits
    )
    _phase2_trackingNtupleSequence = trackingNtupleSequence.copy()
    _phase2_trackingNtupleSequence.remove(siStripMatchedRecHits)
    _phase2_trackingNtupleSequence += (siPhase2RecHits)
    trackingPhase2PU140.toReplaceWith(trackingNtupleSequence, _phase2_trackingNtupleSequence)

if _includeSeeds:
    trackingNtupleSequence += trackingNtupleSeedSelectors

trackingNtupleSequence += (
    # sim information
    trackingParticlesIntime +
    simHitTPAssocProducer +
    tpClusterProducer +
    quickTrackAssociatorByHits +
    trackingParticleNumberOfLayersProducer +
    # ntuplizer
    trackingNtuple
)

trackingPhase2PU140.toModify(trackingNtuple, # FIXME
  pixelDigiSimLink = cms.untracked.InputTag('simSiPixelDigis', "Pixel"),
  stripDigiSimLink = cms.untracked.InputTag(''),
  phase2OTSimLink = cms.untracked.InputTag('simSiPixelDigis', "Tracker")
)
