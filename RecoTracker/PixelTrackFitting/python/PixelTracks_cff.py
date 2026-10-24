import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import *

from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
import RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi
myTTRHBuilderWithoutAngle = RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi.ttrhbwr.clone(
    StripCPE = 'Fake',
    ComponentName = 'PixelTTRHBuilderWithoutAngle'
)
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *
from RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4PixelTriplets_cfi import *
from RecoTracker.PixelTrackFitting.pixelFitterByHelixProjections_cfi import pixelFitterByHelixProjections
from RecoTracker.PixelTrackFitting.pixelNtupletsFitter_cfi import pixelNtupletsFitter
from RecoTracker.PixelTrackFitting.pixelTrackFilterByKinematics_cfi import pixelTrackFilterByKinematics
from RecoTracker.PixelTrackFitting.pixelTrackCleanerBySharedHits_cfi import pixelTrackCleanerBySharedHits
from RecoTracker.PixelTrackFitting.pixelTracks_cfi import pixelTracks as _pixelTracks
from RecoTracker.TkTrackingRegions.globalTrackingRegion_cfi import globalTrackingRegion as _globalTrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpot_cfi import globalTrackingRegionFromBeamSpot as _globalTrackingRegionFromBeamSpot
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
from RecoTracker.PixelSeeding.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoTracker.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoTracker.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder

# Eras
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
from Configuration.Eras.Modifier_run3_common_cff import run3_common

# seeding layers
from RecoTracker.IterativeTracking.InitialStep_cff import initialStepSeedLayers, initialStepHitDoublets, _initialStepCAHitQuadruplets

# TrackingRegion
pixelTracksTrackingRegions = _globalTrackingRegion.clone()
trackingLowPU.toReplaceWith(pixelTracksTrackingRegions, _globalTrackingRegionFromBeamSpot.clone())


# Pixel quadruplets tracking
pixelTracksSeedLayers = initialStepSeedLayers.clone(
    BPix = dict(HitProducer = "siPixelRecHitsPreSplitting"),
    FPix = dict(HitProducer = "siPixelRecHitsPreSplitting")
)

pixelTracksHitDoublets = initialStepHitDoublets.clone(
    clusterCheck = "",
    seedingLayers = "pixelTracksSeedLayers",
    trackingRegions = "pixelTracksTrackingRegions"
)

pixelTracksHitQuadruplets = _initialStepCAHitQuadruplets.clone(
    doublets = "pixelTracksHitDoublets",
    SeedComparitorPSet = dict(clusterShapeCacheSrc = 'siPixelClusterShapeCachePreSplitting')
)

pixelTracks = _pixelTracks.clone(
    SeedingHitSets = "pixelTracksHitQuadruplets"
)

pixelTracksTask = cms.Task(
    pixelTracksTrackingRegions,
    pixelFitterByHelixProjections,
    pixelTrackFilterByKinematics,
    pixelTracksSeedLayers,
    pixelTracksHitDoublets,
    pixelTracksHitQuadruplets,
    pixelTracks
)

pixelTracksSequence = cms.Sequence(pixelTracksTask)


# Pixel triplets for trackingLowPU
pixelTracksHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets = "pixelTracksHitDoublets",
    produceSeedingHitSets = True,
    SeedComparitorPSet = RecoTracker.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone(
        clusterShapeCacheSrc = "siPixelClusterShapeCachePreSplitting"
    )
)

trackingLowPU.toModify(pixelTracks,
    SeedingHitSets = "pixelTracksHitTriplets"
)

_pixelTracksTask_lowPU = pixelTracksTask.copy()
_pixelTracksTask_lowPU.replace(pixelTracksHitQuadruplets, pixelTracksHitTriplets)
trackingLowPU.toReplaceWith(pixelTracksTask, _pixelTracksTask_lowPU)

# Phase 2 modifier
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
# HIon modifiers
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA

######################################################################

### Alpaka Pixel Track Reco

from Configuration.ProcessModifiers.alpaka_cff import alpaka

# pixel tracks SoA producer on the device
from RecoTracker.PixelSeeding.caHitNtupletAlpakaPhase1_cfi import caHitNtupletAlpakaPhase1 as _pixelTracksAlpakaPhase1
from RecoTracker.PixelSeeding.caHitNtupletAlpakaPhase2_cfi import caHitNtupletAlpakaPhase2 as _pixelTracksAlpakaPhase2
from RecoTracker.PixelSeeding.caHitNtupletAlpakaHIonPhase1_cfi import caHitNtupletAlpakaHIonPhase1 as _pixelTracksAlpakaHIonPhase1
from RecoTracker.PixelSeeding.caHitNtupletAlpakaPhase2OT_cfi import caHitNtupletAlpakaPhase2OT as _pixelTracksAlpakaPhase2Extended

pixelTracksAlpaka = _pixelTracksAlpakaPhase1.clone(
    avgHitsPerTrack    = 4.6,      
    avgCellsPerHit     = 13,
    avgCellsPerCell    = 0.0268, 
    avgTracksPerCell   = 0.0123, 
    maxNumberOfDoublets = str(512*1024),    # could be lowered to 315k, keeping the same for a fair comparison with master
    maxNumberOfTuples   = str(32 * 1024),   # this couul be much lower (2.1k, these are quads)
)

phase2_tracker.toReplaceWith(pixelTracksAlpaka,_pixelTracksAlpakaPhase2.clone())

def _modifyForPPonAAandNotPhase2(producer):
    nPairs = int(len(producer.geometry.pairGraph) / 2)
    producer.maxNumberOfDoublets = str(6 * 512 *1024)    # this could be 2.3M
    producer.maxNumberOfTuples = str(256 * 1024)         # this could be 4.7
    producer.avgHitsPerTrack = 5.0
    producer.avgCellsPerHit = 40
    producer.avgCellsPerCell = 0.07                      # with maxNumberOfDoublets ~= 3.14M; 0.02  for HLT HI on 2024 HI Data 
    producer.avgTracksPerCell = 0.03                     # with maxNumberOfDoublets ~= 3.14M; 0.005 for HLT HI on 2024 HI Data
    producer.cellZ0Cut = 8.0                             # setup currenlty used @ HLT (was 10.0) 
    producer.geometry.ptCuts = [0.5] * nPairs            # setup currenlty used @ HLT (was 0.0) 

(pp_on_AA & ~phase2_tracker).toModify(pixelTracksAlpaka, _modifyForPPonAAandNotPhase2)

from Configuration.ProcessModifiers.phase2CAExtension_cff import phase2CAExtension
phase2CAExtension.toReplaceWith(pixelTracksAlpaka,_pixelTracksAlpakaPhase2Extended.clone(
    pixelRecHitSrc = "siPixelRecHitsPreSplittingAlpaka",
    trackerRecHitsSoA = "phase2OTRecHitsSoAConverter"
))

# Phase-2 CA extended to outer-tracker vector hits: the outer-tracker hits in SoA format, the vector hits built from them
# and the vector hit CA, on the device and on the cpu
from Configuration.ProcessModifiers.phase2CAVectorHits_cff import phase2CAVectorHits
from RecoTracker.PixelSeeding.pixelSeedingOTRecHitsSoAConverter_cfi import pixelSeedingOTRecHitsSoAConverter as _pixelSeedingOTRecHitsSoAConverter
from RecoTracker.PixelSeeding.phase2OTVectorHitsSoAProducer_cfi import phase2OTVectorHitsSoAProducer as _phase2OTVectorHitsSoAProducer
from RecoTracker.PixelSeeding.caHitNtupletAlpakaPhase2OTVectorHits_cfi import caHitNtupletAlpakaPhase2OTVectorHits as _pixelTracksAlpakaPhase2OTVectorHits
from RecoTracker.FinalTrackSelectors.pixelTrackForestHighPuritySelector_cfi import pixelTrackForestHighPuritySelector as _pixelTrackForestHighPuritySelector

pixelSeedingOTRecHitsSoA = _pixelSeedingOTRecHitsSoAConverter.clone()
pixelSeedingOTRecHitsSoASerial = makeSerialClone(pixelSeedingOTRecHitsSoA)
phase2OTVectorHitsSoA = _phase2OTVectorHitsSoAProducer.clone(
    otRecHitsSoA = "pixelSeedingOTRecHitsSoA"
)
phase2OTVectorHitsSoASerial = makeSerialClone(phase2OTVectorHitsSoA,
    otRecHitsSoA = "pixelSeedingOTRecHitsSoASerial"
)

phase2CAVectorHits.toReplaceWith(pixelTracksAlpaka, _pixelTracksAlpakaPhase2OTVectorHits.clone(
    pixelRecHitSrc = "siPixelRecHitsPreSplittingAlpaka",
    vectorHitsSrc = "phase2OTVectorHitsSoA"
))

# pixel tracks SoA producer on the cpu, for validation
pixelTracksAlpakaSerial = makeSerialClone(pixelTracksAlpaka,
    pixelRecHitSrc = 'siPixelRecHitsPreSplittingAlpakaSerial'
)

phase2CAExtension.toModify(pixelTracksAlpakaSerial,
                           pixelRecHitSrc = 'siPixelRecHitsExtendedPreSplittingAlpakaSerial'
                           )

phase2CAVectorHits.toModify(pixelTracksAlpakaSerial,
    vectorHitsSrc = "phase2OTVectorHitsSoASerial"
)

# high-purity selection of the vector hit CA tracks, as in the HLT menu
pixelTracksAlpakaHighPurity = _pixelTrackForestHighPuritySelector.clone(
    pixelTrackSrc = "pixelTracksAlpaka",
    pixelRecHitSrc = "siPixelRecHitsPreSplittingAlpaka",
    vectorHitsSrc = "phase2OTVectorHitsSoA",
    otRecHitsSoASrc = "pixelSeedingOTRecHitsSoA",
    model = "RecoTracker/FinalTrackSelectors/data/PixelTrackTorchHighPuritySelector/prompt_tree31_wp_20260914.bin"
)

# legacy pixel tracks from SoA
from  RecoTracker.PixelTrackFitting.pixelTrackProducerFromSoAAlpaka_cfi import pixelTrackProducerFromSoAAlpaka as _pixelTrackProducerFromSoAAlpaka

(alpaka & ~phase2CAExtension).toReplaceWith(pixelTracks, _pixelTrackProducerFromSoAAlpaka.clone(
    pixelRecHitLegacySrc = "siPixelRecHitsPreSplitting",
))

phase2CAExtension.toReplaceWith(pixelTracks, _pixelTrackProducerFromSoAAlpaka.clone(
    pixelRecHitLegacySrc = "siPixelRecHitsPreSplitting",
    beamSpot = cms.InputTag("offlineBeamSpot"),
    minNumberOfHits = cms.int32(0),
    minQuality = cms.string('tight'),
    trackSrc = cms.InputTag("pixelTracksAlpaka"),
    outerTrackerRecHitSrc = cms.InputTag("siPhase2RecHits"),
    outerTrackerRecHitSoAConverterSrc = cms.InputTag("phase2OTRecHitsSoAConverter"),
    useOTExtension = cms.bool(True),
    requireQuadsFromConsecutiveLayers = cms.bool(True)
))

# the vector hits are expanded back into their two outer-tracker rechits
phase2CAVectorHits.toReplaceWith(pixelTracks, _pixelTrackProducerFromSoAAlpaka.clone(
    trackSrc = "pixelTracksAlpakaHighPurity",
    pixelRecHitLegacySrc = "siPixelRecHitsPreSplitting",
    outerTrackerRecHitSrc = "siPhase2RecHits",
    otRecHitsSoASrc = "pixelSeedingOTRecHitsSoA",
    vectorHitsSoASrc = "phase2OTVectorHitsSoA",
    minQuality = "tight",
    useOTExtension = True,
    expandVectorHits = True
))

alpaka.toReplaceWith(pixelTracksTask, cms.Task(
    # Build the pixel ntuplets and the pixel tracks in SoA format with alpaka on the device
    pixelTracksAlpaka,
    # Build the pixel ntuplets and the pixel tracks in SoA format with alpaka on the cpu (if requested by the validation)
    pixelTracksAlpakaSerial,
    # Convert the pixel tracks from SoA to legacy format
    pixelTracks)
)

phase2CAVectorHits.toReplaceWith(pixelTracksTask, cms.Task(
    pixelSeedingOTRecHitsSoA,
    pixelSeedingOTRecHitsSoASerial,
    phase2OTVectorHitsSoA,
    phase2OTVectorHitsSoASerial,
    pixelTracksAlpaka,
    pixelTracksAlpakaSerial,
    pixelTracksAlpakaHighPurity,
    pixelTracks)
)

# event setup of the vector hit CA: stacked-module geometry
def _addProcessESPhase2CAVectorHits(process):
    process.load("RecoTracker.PixelSeeding.stackedModuleGeometryESProducer_cfi")

modifyConfigurationForPhase2CAVectorHits_ = (alpaka & phase2CAVectorHits).makeProcessModifier(_addProcessESPhase2CAVectorHits)
