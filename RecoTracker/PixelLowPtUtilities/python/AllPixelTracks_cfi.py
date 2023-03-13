import FWCore.ParameterSet.Config as cms

from RecoTracker.PixelLowPtUtilities.clusterShapeTrackFilter_cfi import clusterShapeTrackFilter
from RecoTracker.PixelLowPtUtilities.trackFitter_cfi import trackFitter
from RecoTracker.PixelLowPtUtilities.trackCleaner_cfi import trackCleaner
import RecoTracker.PixelTrackFitting.pixelTracks_cfi as _mod

##########################
# The base for all steps
allPixelTracks = _mod.pixelTracks.clone(
    passLabel  = '',
    # Region
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalTrackingRegionWithVerticesProducer'),
        RegionPSet = cms.PSet(
            precise       = cms.bool(True),
            beamSpot      = cms.InputTag("offlineBeamSpot"),
            originRadius  = cms.double(0.2),
            sigmaZVertex  = cms.double(3.0),
            useFixedError = cms.bool(True),
            fixedError    = cms.double(0.2),

            useFoundVertices = cms.bool(False),
            VertexCollection = cms.InputTag("pixel3Vertices"),
            ptMin            = cms.double(0.075),
            nSigmaZ          = cms.double(3.0)
        )
    ),
    # Ordered hits
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        SeedingLayers = cms.InputTag('PixelLayerTriplets'),
        GeneratorPSet = cms.PSet(
            ComponentName = cms.string('PixelTripletLowPtGenerator'),
            checkClusterShape       = cms.bool(False),
            checkMultipleScattering = cms.bool(True),
            nSigMultipleScattering  = cms.double(5.0),
            maxAngleRatio = cms.double(10.0),
            rzTolerance   = cms.double(0.2),
            TTRHBuilder   = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
            clusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache")
        )
    ),

    # Filter
    Filter = "clusterShapeTrackFilter",

    # Cleaner
    CleanerPSet = cms.string("trackCleaner"),

    # Fitter
    FitterPSet = cms.InputTag("allPixelTracksFitter"),
)

allPixelTracksFitter = trackFitter.clone(
    TTRHBuilder = 'TTRHBuilderWithoutAngle4PixelTriplets'
)

allPixelTracksTask = cms.Task(
    allPixelTracksFitter,
    clusterShapeTrackFilter,
    allPixelTracks
)
allPixelTracksSequence = cms.Sequence(allPixelTracksTask)
