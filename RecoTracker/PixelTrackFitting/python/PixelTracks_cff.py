import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import *
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

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


# "Patatrack" pixel ntuplets, fishbone cleaning, Broken Line fit, and density-based vertex reconstruction
from Configuration.ProcessModifiers.pixelNtupletFit_cff import pixelNtupletFit
from RecoTracker.PixelSeeding.caHitNtupletCUDAPhase1_cfi import caHitNtupletCUDAPhase1 as _pixelTracksCUDA
from RecoTracker.PixelSeeding.caHitNtupletCUDAPhase2_cfi import caHitNtupletCUDAPhase2 as _pixelTracksCUDAPhase2
from RecoTracker.PixelSeeding.caHitNtupletCUDAHIonPhase1_cfi import caHitNtupletCUDAHIonPhase1 as _pixelTracksCUDAHIonPhase1
# Phase 2 modifier
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
# HIon modifiers
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA

# SwitchProducer providing the pixel tracks in SoA format on the CPU
pixelTracksSoA = SwitchProducerCUDA(
    # build pixel ntuplets and pixel tracks in SoA format on the CPU
    cpu = _pixelTracksCUDA.clone(
        pixelRecHitSrc = "siPixelRecHitsPreSplittingSoA",
        idealConditions = False,
        onGPU = False
    )
)

# use quality cuts tuned for Run 2 ideal conditions for all Run 3 workflows
run3_common.toModify(pixelTracksSoA.cpu,
    idealConditions = True
)

# convert the pixel tracks from SoA to legacy format
from RecoTracker.PixelTrackFitting.pixelTrackProducerFromSoAPhase1_cfi import pixelTrackProducerFromSoAPhase1 as _pixelTrackProducerFromSoA
from RecoTracker.PixelTrackFitting.pixelTrackProducerFromSoAPhase2_cfi import pixelTrackProducerFromSoAPhase2 as _pixelTrackProducerFromSoAPhase2
from RecoTracker.PixelTrackFitting.pixelTrackProducerFromSoAHIonPhase1_cfi import pixelTrackProducerFromSoAHIonPhase1 as _pixelTrackProducerFromSoAHIonPhase1

pixelNtupletFit.toReplaceWith(pixelTracks, _pixelTrackProducerFromSoA.clone(
    pixelRecHitLegacySrc = "siPixelRecHitsPreSplitting",
))

(pixelNtupletFit & phase2_tracker).toReplaceWith(pixelTracks, _pixelTrackProducerFromSoAPhase2.clone(
    pixelRecHitLegacySrc = "siPixelRecHitsPreSplitting",
))

(pixelNtupletFit & pp_on_AA).toReplaceWith(pixelTracks, _pixelTrackProducerFromSoAHIonPhase1.clone(
    pixelRecHitLegacySrc = "siPixelRecHitsPreSplitting",
))

pixelNtupletFit.toReplaceWith(pixelTracksTask, cms.Task(
    # build the pixel ntuplets and the pixel tracks in SoA format on the GPU
    pixelTracksSoA,
    # convert the pixel tracks from SoA to legacy format
    pixelTracks
))

# "Patatrack" sequence running on GPU (or CPU if not available)
from Configuration.ProcessModifiers.gpu_cff import gpu

# build the pixel ntuplets and pixel tracks in SoA format on the GPU
pixelTracksCUDA = _pixelTracksCUDA.clone(
    pixelRecHitSrc = "siPixelRecHitsPreSplittingCUDA",
    idealConditions = False,
    onGPU = True,
)

# use quality cuts tuned for Run 2 ideal conditions for all Run 3 workflows
run3_common.toModify(pixelTracksCUDA,
    idealConditions = True
)

# SwitchProducer providing the pixel tracks in SoA format on the CPU
from RecoTracker.PixelTrackFitting.pixelTrackSoAFromCUDAPhase1_cfi import pixelTrackSoAFromCUDAPhase1 as _pixelTracksSoA
from RecoTracker.PixelTrackFitting.pixelTrackSoAFromCUDAPhase2_cfi import pixelTrackSoAFromCUDAPhase2 as _pixelTracksSoAPhase2
from RecoTracker.PixelTrackFitting.pixelTrackSoAFromCUDAHIonPhase1_cfi import pixelTrackSoAFromCUDAHIonPhase1 as _pixelTracksSoAHIonPhase1

gpu.toModify(pixelTracksSoA,
    # transfer the pixel tracks in SoA format to the host
    cuda = _pixelTracksSoA.clone()
)

(gpu & phase2_tracker).toModify(pixelTracksSoA,cuda = _pixelTracksSoAPhase2.clone(
))

(gpu & pp_on_AA).toModify(pixelTracksSoA,cuda = _pixelTracksSoAHIonPhase1.clone(
))

phase2_tracker.toModify(pixelTracksSoA,cpu = _pixelTracksCUDAPhase2.clone(
    pixelRecHitSrc = "siPixelRecHitsPreSplittingSoA",
    onGPU = False
))

(pp_on_AA & ~phase2_tracker).toModify(pixelTracksSoA,cpu = _pixelTracksCUDAHIonPhase1.clone(
    pixelRecHitSrc = "siPixelRecHitsPreSplittingSoA",
    onGPU = False
))

phase2_tracker.toReplaceWith(pixelTracksCUDA,_pixelTracksCUDAPhase2.clone(
    pixelRecHitSrc = "siPixelRecHitsPreSplittingCUDA",
    onGPU = True,
))

(pp_on_AA & ~phase2_tracker).toReplaceWith(pixelTracksCUDA,_pixelTracksCUDAHIonPhase1.clone(
    pixelRecHitSrc = "siPixelRecHitsPreSplittingCUDA",
    onGPU = True,
))

(pixelNtupletFit & gpu).toReplaceWith(pixelTracksTask, cms.Task(
    # build the pixel ntuplets and pixel tracks in SoA format on the GPU
    pixelTracksCUDA,
    # transfer the pixel tracks in SoA format to the CPU, and convert them to legacy format
    pixelTracksTask.copy()
))

## GPU vs CPU validation
# force CPU vertexing to use hit SoA from CPU chain and not the converted one from GPU chain
from Configuration.ProcessModifiers.gpuValidationPixel_cff import gpuValidationPixel
(pixelNtupletFit & gpu & gpuValidationPixel).toModify(pixelTracksSoA.cpu,
    pixelRecHitSrc = "siPixelRecHitsPreSplittingSoA@cpu"
    )

######################################################################

### Alpaka Pixel Track Reco

from Configuration.ProcessModifiers.alpaka_cff import alpaka

# pixel tracks SoA producer on the device
from RecoTracker.PixelSeeding.caHitNtupletAlpakaPhase1_cfi import caHitNtupletAlpakaPhase1 as _pixelTracksAlpakaPhase1
from RecoTracker.PixelSeeding.caHitNtupletAlpakaPhase2_cfi import caHitNtupletAlpakaPhase2 as _pixelTracksAlpakaPhase2
from RecoTracker.PixelSeeding.caHitNtupletAlpakaHIonPhase1_cfi import caHitNtupletAlpakaHIonPhase1 as _pixelTracksAlpakaHIonPhase1

pixelTracksAlpaka = _pixelTracksAlpakaPhase1.clone()
phase2_tracker.toReplaceWith(pixelTracksAlpaka,_pixelTracksAlpakaPhase2.clone())
(pp_on_AA & ~phase2_tracker).toReplaceWith(pixelTracksAlpaka, _pixelTracksAlpakaHIonPhase1.clone())

# pixel tracks SoA producer on the cpu, for validation
pixelTracksAlpakaSerial = makeSerialClone(pixelTracksAlpaka,
    pixelRecHitSrc = 'siPixelRecHitsPreSplittingAlpakaSerial'
)

# legacy pixel tracks from SoA
from  RecoTracker.PixelTrackFitting.pixelTrackProducerFromSoAAlpakaPhase1_cfi import pixelTrackProducerFromSoAAlpakaPhase1 as _pixelTrackProducerFromSoAAlpakaPhase1
from  RecoTracker.PixelTrackFitting.pixelTrackProducerFromSoAAlpakaPhase2_cfi import pixelTrackProducerFromSoAAlpakaPhase2 as _pixelTrackProducerFromSoAAlpakaPhase2
from  RecoTracker.PixelTrackFitting.pixelTrackProducerFromSoAAlpakaHIonPhase1_cfi import pixelTrackProducerFromSoAAlpakaHIonPhase1 as _pixelTrackProducerFromSoAAlpakaHIonPhase1

(alpaka & ~phase2_tracker).toReplaceWith(pixelTracks, _pixelTrackProducerFromSoAAlpakaPhase1.clone(
    pixelRecHitLegacySrc = "siPixelRecHitsPreSplitting",
))

(alpaka & phase2_tracker).toReplaceWith(pixelTracks, _pixelTrackProducerFromSoAAlpakaPhase2.clone(
    pixelRecHitLegacySrc = "siPixelRecHitsPreSplitting",
))

(alpaka & ~phase2_tracker & pp_on_AA).toReplaceWith(pixelTracks, _pixelTrackProducerFromSoAAlpakaHIonPhase1.clone(
    pixelRecHitLegacySrc = "siPixelRecHitsPreSplitting",
))

alpaka.toReplaceWith(pixelTracksTask, cms.Task(
    # Build the pixel ntuplets and the pixel tracks in SoA format with alpaka on the device
    pixelTracksAlpaka,
    # Build the pixel ntuplets and the pixel tracks in SoA format with alpaka on the cpu (if requested by the validation)
    pixelTracksAlpakaSerial,
    # Convert the pixel tracks from SoA to legacy format
    pixelTracks)
)

# Patatrack with strip hits (alpaka-only)

from Configuration.ProcessModifiers.stripNtupletFit_cff import stripNtupletFit
from RecoLocalTracker.Configuration.RecoLocalTracker_cff import striptrackerlocalrecoTask
from RecoLocalTracker.SiStripRecHitConverter.siStripRecHitSoAPhase1_cfi import siStripRecHitSoAPhase1
from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import quickTrackAssociatorByHits, quickTrackAssociatorByHitsTrackerHitAssociator
from Validation.RecoTrack.TrackValidation_cff import quickTrackAssociatorByHitsPreSplitting
from RecoTracker.PixelTrackFitting.pixelTrackProducerFromSoAAlpakaPhase1Strip_cfi import pixelTrackProducerFromSoAAlpakaPhase1Strip as _pixelTrackProducerFromSoAAlpakaPhase1Strip

from RecoTracker.PixelSeeding.caHitNtupletAlpakaPhase1Strip_cfi import caHitNtupletAlpakaPhase1Strip as _pixelTracksAlpakaPhase1Strip

(alpaka & stripNtupletFit & ~phase2_tracker).toReplaceWith(pixelTracks, _pixelTrackProducerFromSoAAlpakaPhase1Strip.clone(useStripHits = cms.bool(True), minQuality = cms.string('loose'),hitModuleStartSrc = cms.InputTag("siStripRecHitSoAPhase1")))

(alpaka & stripNtupletFit & ~phase2_tracker).toReplaceWith(pixelTracksAlpaka, _pixelTracksAlpakaPhase1Strip.clone(pixelRecHitSrc = cms.InputTag("siStripRecHitSoAPhase1")))

siStripRecHitSoAPhase1Serial = makeSerialClone(siStripRecHitSoAPhase1)

(alpaka & stripNtupletFit & ~phase2_tracker).toReplaceWith(pixelTracksAlpakaSerial,makeSerialClone(_pixelTracksAlpakaPhase1Strip,
    pixelRecHitSrc = 'siStripRecHitSoAPhase1Serial'
))

(alpaka & stripNtupletFit & ~phase2_tracker).toReplaceWith(quickTrackAssociatorByHits, quickTrackAssociatorByHitsTrackerHitAssociator.clone(cluster2TPSrc = "tpClusterProducerPreSplitting"))

(alpaka & stripNtupletFit & ~phase2_tracker).toReplaceWith(pixelTracksTask, cms.Task(
    # build legacy strip hits
    striptrackerlocalrecoTask,
    # mix pixel and strip hits in a SoA
    siStripRecHitSoAPhase1,
    siStripRecHitSoAPhase1Serial,
    # Build the pixel ntuplets and the pixel tracks in SoA format with alpaka on the device
    pixelTracksAlpaka,
    # # Build the pixel ntuplets and the pixel tracks in SoA format with alpaka on the cpu (if requested by the validation)
    pixelTracksAlpakaSerial,
    # Convert the pixel tracks from SoA to legacy format
    pixelTracks
))

# ### Alpaka Device vs Host validation

# from Configuration.ProcessModifiers.alpakaValidationPixel_cff import alpakaValidationPixel

# # Hit SoA producer on serial backend
# pixelTracksAlpakaSerial = pixelTracksAlpaka.clone(
#     pixelRecHitSrc = 'siPixelRecHitsPreSplittingAlpakaSerial',
#     alpaka = dict( backend = 'serial_sync' )
# )

# siStripRecHitSoAPhase1Serial = siStripRecHitSoAPhase1.clone(
#     pixelRecHitSoASource = cms.InputTag('siPixelRecHitsPreSplittingAlpakaSerial'),
#     alpaka = dict( backend = 'serial_sync' )
# )

# (alpakaValidationPixel & stripNtupletFit & ~phase2_tracker).toModify(pixelTracksAlpakaSerial,
#     pixelRecHitSrc = 'siStripRecHitSoAPhase1Serial'
# )

# (alpakaValidationPixel & ~stripNtupletFit).toReplaceWith(pixelTracksTask, cms.Task(
#                         # Reconstruct and convert the pixel tracks with alpaka on device
#                         pixelTracksTask.copy(),
#                         # SoA serial counterpart
#                         pixelTracksAlpakaSerial))

# (alpakaValidationPixel & stripNtupletFit).toReplaceWith(pixelTracksTask, cms.Task(
#                         # Reconstruct and convert the pixel tracks with alpaka on device
#                         pixelTracksTask.copy(),
#                         # mix pixel and strips serial
#                         siStripRecHitSoAPhase1Serial,
#                         # SoA serial counterpart
#                         pixelTracksAlpakaSerial))

