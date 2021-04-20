from __future__ import absolute_import
import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi import *
from SimTracker.VertexAssociation.VertexAssociatorByPositionAndTracks_cfi import *
import Validation.RecoTrack.cutsRecoTracks_cfi as cutsRecoTracks_cfi

quality = {
    "L"  : "loose",
    "T"  : "tight",
    "HP" : "highPurity",
}

### Pixel tracking only mode (placeholder for now)
trackingParticlePixelTrackAsssociation = trackingParticleRecoTrackAsssociation.clone(
    label_tr = "pixelTracks",
    associator = "quickTrackAssociatorByHitsPreSplitting",
)
PixelVertexAssociatorByPositionAndTracks = VertexAssociatorByPositionAndTracks.clone(
    trackAssociation = "trackingParticlePixelTrackAsssociation"
)

_pixelTracksCustom = dict(
    src = "pixelTracks",
    vertexTag = "pixelVertices",
)

from CommonTools.RecoAlgos.TrackWithVertexRefSelector_cfi import trackWithVertexRefSelector as _trackWithVertexRefSelector
pixelTracksFromPV = _trackWithVertexRefSelector.clone(
    src = "pixelTracks",
    ptMin = 0,
    ptMax = 1e10,
    ptErrorCut = 1e10,
    quality = "loose",
    vertexTag = "pixelVertices",
    nVertices = 1,
    vtxFallback = False,
    zetaVtx = 0.1, # 1 mm
    rhoVtx = 1e10, # intentionally no dxy cut
)

### pixelTracks loose, tight, highPurity
for key,value in quality.items():
    label = "pixelTracks"+str(key)
    print label
    locals()[label] = cutsRecoTracks_cfi.cutsRecoTracks.clone(
        quality = [value], **_pixelTracksCustom) ## quality    

    label = "pixelTracksPt09"+key
    print label
    locals()[label] = cutsRecoTracks_cfi.cutsRecoTracks.clone(
        ptMin=0.9, quality = [value], **_pixelTracksCustom) ## quality

    labelFromPV = "pixelTracksFromPVPt09"+key
    print labelFromPV
    locals()[labelFromPV] = locals()[label].clone(src = "pixelTracksFromPV")

    label = "pixelTracks4hits"+key
    print label
    locals()[label] = cutsRecoTracks_cfi.cutsRecoTracks.clone(
        minHit = 4, quality = [value], **_pixelTracksCustom) ## quality

    labelFromPV = "pixelTracksFromPV4hits"+key
    print labelFromPV
    locals()[labelFromPV] = locals()[label].clone(src = "pixelTracksFromPV")


trackSelector = cms.EDFilter('TrackSelector',
                             src = cms.InputTag('pixelTracks'),
                             cut = cms.string("")
)
for key,value in quality.items():
    label = "pixelTracks3hits"+key
    print label
    cutstring = "numberOfValidHits == 3 & quality('" + value + "')" 
    print cutstring
    locals()[label] = trackSelector.clone( cut = cutstring )

    labelFromPV = "pixelTracksFromPV4hits"+key
    print labelFromPV
    locals()[labelFromPV] = locals()[label].clone(src = "pixelTracksFromPV")

import Validation.RecoTrack.MultiTrackValidator_cfi
trackValidatorPixelTrackingOnly = Validation.RecoTrack.MultiTrackValidator_cfi.multiTrackValidator.clone(
    useLogPt = cms.untracked.bool(True),
    doPVAssociationPlots = True,
    dirName = "Tracking/PixelTrack/",
    label = [
        "pixelTracksL", "pixelTracksPt09L", "pixelTracks3hitsL", "pixelTracks4hitsL",
        "pixelTracksT", "pixelTracksPt09T", "pixelTracks3hitsT", "pixelTracks4hitsT",
        "pixelTracksHP", "pixelTracksPt09HP", "pixelTracks3hitsHP", "pixelTracks4hitsHP",
         ],
    doResolutionPlotsForLabels = [],
    trackCollectionForDrCalculation = "pixelTracks",
    associators = ["trackingParticlePixelTrackAsssociation"],
    label_vertex = "pixelVertices",
    vertexAssociator = "PixelVertexAssociatorByPositionAndTracks",
    dodEdxPlots = False,
    cores = cms.InputTag(""),
)
print trackValidatorPixelTrackingOnly.label

trackValidatorFromPVPixelTrackingOnly = trackValidatorPixelTrackingOnly.clone(
    dirName = "Tracking/PixelTrackFromPV/",
    label = [
        "pixelTracksFromPVL", "pixelTracksFromPVPt09L", "pixelTracksFromPV3hitsL", "pixelTracksFromPV4hitsL"
        "pixelTracksFromPVT", "pixelTracksFromPVPt09T", "pixelTracksFromPV3hitsT", "pixelTracksFromPV4hitsT"
        "pixelTracksFromPVHP", "pixelTracksFromPVPt09HP", "pixelTracksFromPV3hitsHP", "pixelTracksFromPV4hitsHP"
    ],
    label_tp_effic = "trackingParticlesSignal",
    label_tp_fake = "trackingParticlesSignal",
    label_tp_effic_refvector = True,
    label_tp_fake_refvector = True,
    trackCollectionForDrCalculation = "pixelTracksFromPV",
    doPlotsOnlyForTruePV = True,
    doPVAssociationPlots = False,
    doResolutionPlotsForLabels = ["disabled"],
)
trackValidatorFromPVAllTPPixelTrackingOnly = trackValidatorFromPVPixelTrackingOnly.clone(
    dirName = "Tracking/PixelTrackFromPVAllTP/",
    label_tp_effic = trackValidatorPixelTrackingOnly.label_tp_effic.value(),
    label_tp_fake = trackValidatorPixelTrackingOnly.label_tp_fake.value(),
    label_tp_effic_refvector = False,
    label_tp_fake_refvector = False,
    doSimPlots = False,
    doSimTrackPlots = False,
)
trackValidatorBHadronPixelTrackingOnly = trackValidatorPixelTrackingOnly.clone(
    dirName = "Tracking/PixelTrackBHadron/",
    label = [
        "pixelTracksL", "pixelTracks3hitsL", "pixelTracks4hitsL",
        "pixelTracksT", "pixelTracks3hitsT", "pixelTracks4hitsT",
        "pixelTracksHP", "pixelTracks3hitsHP", "pixelTracks4hitsHP",
         ],
    label_tp_effic = "trackingParticlesBHadron",
    label_tp_effic_refvector = True,
    doSimPlots = True,
    doRecoTrackPlots = False, # Fake rate is defined wrt. all TPs, and that is already included in trackValidator
    dodEdxPlots = False,
)

from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import *
from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import *
from SimGeneral.TrackingAnalysis.trackingParticleNumberOfLayersProducer_cff import *

# Select B-hadron TPs
from SimTracker.TrackHistory.trackingParticleBHadronRefSelector_cfi import trackingParticleBHadronRefSelector as _trackingParticleBHadronRefSelector
trackingParticlesBHadron = _trackingParticleBHadronRefSelector.clone()

tracksValidationTruthPixelTrackingOnly = cms.Task(
    tpClusterProducer,
    quickTrackAssociatorByHits,
    trackingParticlePixelTrackAsssociation,
    PixelVertexAssociatorByPositionAndTracks,
    trackingParticleNumberOfLayersProducer,
    trackingParticlesBHadron
)

from CommonTools.RecoAlgos.trackingParticleRefSelector_cfi import trackingParticleRefSelector as _trackingParticleRefSelector
## Select signal TrackingParticles, and do the corresponding associations
trackingParticlesSignal = _trackingParticleRefSelector.clone(
    signalOnly = True,
    chargedOnly = False,
    tip = 1e5,
    lip = 1e5,
    minRapidity = -10,
    maxRapidity = 10,
    ptMin = 0,
)
tracksPreValidationPixelTrackingOnly = cms.Task(
    tracksValidationTruthPixelTrackingOnly,
    trackingParticlesSignal,
)

for category in ["pixelTracks","pixelTracksPt09","pixelTracksFromPVPt09","pixelTracks4hits","pixelTracksFromPV4hits","pixelTracks3hits","pixelTracksFromPV4hits"]:    
#for category in ["pixelTracks"]:
    for key in quality:
        label = category+key
#        label = "pixelTracks"+key
        print label
        tracksPreValidationPixelTrackingOnly.add(locals()[label])

tracksValidationPixelTrackingOnly = cms.Sequence(
    trackValidatorPixelTrackingOnly +
    trackValidatorFromPVPixelTrackingOnly +
    trackValidatorFromPVAllTPPixelTrackingOnly +
    trackValidatorBHadronPixelTrackingOnly,
    tracksPreValidationPixelTrackingOnly
)
