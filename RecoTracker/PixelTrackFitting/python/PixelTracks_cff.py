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

# Phase-2 CA extended to outer-tracker stubs: the outer-tracker hits in SoA format, the stubs built from them,
# the pixel hits merged with the stubs and the stub CA, on the device and on the cpu
from Configuration.ProcessModifiers.phase2CAStubs_cff import phase2CAStubs
from RecoTracker.PixelSeeding.pixelSeedingOTRecHitsSoAConverter_cfi import pixelSeedingOTRecHitsSoAConverter as _pixelSeedingOTRecHitsSoAConverter
from RecoTracker.PixelSeeding.otStubProducerVectorHitStyle_cfi import otStubProducerVectorHitStyle as _otStubProducerVectorHitStyle
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitsStubsMerger_cfi import siPixelRecHitsStubsMerger as _siPixelRecHitsStubsMerger
from RecoTracker.PixelSeeding.caHitNtupletAlpakaPhase2OTStubs_cfi import caHitNtupletAlpakaPhase2OTStubs as _pixelTracksAlpakaPhase2OTStubs
from RecoTracker.FinalTrackSelectors.pixelTrackForestHighPuritySelector_cfi import pixelTrackForestHighPuritySelector as _pixelTrackForestHighPuritySelector

pixelSeedingOTRecHitsSoA = _pixelSeedingOTRecHitsSoAConverter.clone()
pixelSeedingOTRecHitsSoASerial = makeSerialClone(pixelSeedingOTRecHitsSoA)
otStubProducer = _otStubProducerVectorHitStyle.clone(
    otRecHitsSoA = "pixelSeedingOTRecHitsSoA"
)
otStubProducerSerial = makeSerialClone(otStubProducer,
    otRecHitsSoA = "pixelSeedingOTRecHitsSoASerial"
)
siPixelRecHitsStubsMerger = _siPixelRecHitsStubsMerger.clone(
    pixelRecHitsSoA = "siPixelRecHitsPreSplittingAlpaka",
    stubsSoA = "otStubProducer",
    otRecHitsSoA = "pixelSeedingOTRecHitsSoA"
)
siPixelRecHitsStubsMergerSerial = makeSerialClone(siPixelRecHitsStubsMerger,
    pixelRecHitsSoA = "siPixelRecHitsPreSplittingAlpakaSerial",
    stubsSoA = "otStubProducerSerial",
    otRecHitsSoA = "pixelSeedingOTRecHitsSoASerial"
)

phase2CAStubs.toReplaceWith(pixelTracksAlpaka, _pixelTracksAlpakaPhase2OTStubs.clone(
    pixelRecHitSrc = "siPixelRecHitsStubsMerger"
))

# pixel tracks SoA producer on the cpu, for validation
pixelTracksAlpakaSerial = makeSerialClone(pixelTracksAlpaka,
    pixelRecHitSrc = 'siPixelRecHitsPreSplittingAlpakaSerial'
)

phase2CAExtension.toModify(pixelTracksAlpakaSerial,
                           pixelRecHitSrc = 'siPixelRecHitsExtendedPreSplittingAlpakaSerial'
                           )

phase2CAStubs.toModify(pixelTracksAlpakaSerial,
    pixelRecHitSrc = "siPixelRecHitsStubsMergerSerial"
)

# high-purity selection of the stub CA tracks, as in the HLT menu
pixelTracksAlpakaHighPurity = _pixelTrackForestHighPuritySelector.clone(
    pixelTrackSrc = "pixelTracksAlpaka",
    mergedHitsSrc = "siPixelRecHitsStubsMerger",
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

# the stubs are expanded back into their two outer-tracker rechits
phase2CAStubs.toReplaceWith(pixelTracks, _pixelTrackProducerFromSoAAlpaka.clone(
    trackSrc = "pixelTracksAlpakaHighPurity",
    pixelRecHitLegacySrc = "siPixelRecHitsPreSplitting",
    outerTrackerRecHitSrc = "siPhase2RecHits",
    otRecHitsSoASrc = "pixelSeedingOTRecHitsSoA",
    stubsSoASrc = "otStubProducer",
    minQuality = "tight",
    useOTExtension = True,
    expandStubs = True
))

alpaka.toReplaceWith(pixelTracksTask, cms.Task(
    # Build the pixel ntuplets and the pixel tracks in SoA format with alpaka on the device
    pixelTracksAlpaka,
    # Build the pixel ntuplets and the pixel tracks in SoA format with alpaka on the cpu (if requested by the validation)
    pixelTracksAlpakaSerial,
    # Convert the pixel tracks from SoA to legacy format
    pixelTracks)
)

phase2CAStubs.toReplaceWith(pixelTracksTask, cms.Task(
    pixelSeedingOTRecHitsSoA,
    pixelSeedingOTRecHitsSoASerial,
    otStubProducer,
    otStubProducerSerial,
    siPixelRecHitsStubsMerger,
    siPixelRecHitsStubsMergerSerial,
    pixelTracksAlpaka,
    pixelTracksAlpakaSerial,
    pixelTracksAlpakaHighPurity,
    pixelTracks)
)

# event setup of the stub CA: stacked-module geometry and material map of the BrokenLine fit
def _addProcessESPhase2CAStubs(process):
    process.load("RecoTracker.PixelSeeding.stackedModuleGeometryESProducer_cfi")
    process.load("RecoTracker.PixelTrackFitting.blMaterialMapESProducerAlpaka_cfi")

modifyConfigurationForPhase2CAStubs_ = (alpaka & phase2CAStubs).makeProcessModifier(_addProcessESPhase2CAStubs)
