import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import *
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

from RecoTracker.PixelTrackFitting.PixelTracks_cff import *
from RecoVertex.PixelVertexFinding.PixelVertices_cff import *

# legacy pixel vertex reconsruction using the divisive vertex finder
pixelVerticesTask = cms.Task(
    pixelVertices
)

# "Patatrack" pixel ntuplets, fishbone cleaning, Broken Line fit, and density-based vertex reconstruction
from Configuration.ProcessModifiers.pixelNtupletFit_cff import pixelNtupletFit
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker

# HIon modifiers
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA

# build the pixel vertices in SoA format on the CPU
from RecoVertex.PixelVertexFinding.pixelVertexProducerCUDAPhase1_cfi import pixelVertexProducerCUDAPhase1 as _pixelVerticesCUDA
from RecoVertex.PixelVertexFinding.pixelVertexProducerCUDAPhase2_cfi import pixelVertexProducerCUDAPhase2 as _pixelVerticesCUDAPhase2
from RecoVertex.PixelVertexFinding.pixelVertexProducerCUDAHIonPhase1_cfi import pixelVertexProducerCUDAHIonPhase1 as _pixelVerticesCUDAHIonPhase1

pixelVerticesSoA = SwitchProducerCUDA(
    cpu = _pixelVerticesCUDA.clone(
        pixelTrackSrc = "pixelTracksSoA",
        onGPU = False
    )
)

phase2_tracker.toModify(pixelVerticesSoA,cpu = _pixelVerticesCUDAPhase2.clone(
    pixelTrackSrc = "pixelTracksSoA",
    onGPU = False,
    PtMin = 2.0
))

(pp_on_AA & ~phase2_tracker).toModify(pixelVerticesSoA,cpu = _pixelVerticesCUDAHIonPhase1.clone(
    pixelTrackSrc = "pixelTracksSoA",
    doSplitting = False,
    onGPU = False,
))

# convert the pixel vertices from SoA to legacy format
from RecoVertex.PixelVertexFinding.pixelVertexFromSoA_cfi import pixelVertexFromSoA as _pixelVertexFromSoA

(pixelNtupletFit).toReplaceWith(pixelVertices, _pixelVertexFromSoA.clone(
    src = "pixelVerticesSoA"
))

(pixelNtupletFit).toReplaceWith(pixelVerticesTask, cms.Task(
    # build the pixel vertices in SoA format on the CPU
    pixelVerticesSoA,
    # convert the pixel vertices from SoA to legacy format
    pixelVertices
))


# "Patatrack" sequence running on the GPU
from Configuration.ProcessModifiers.gpu_cff import gpu

# build pixel vertices in SoA format on the GPU
pixelVerticesCUDA = _pixelVerticesCUDA.clone(
    pixelTrackSrc = "pixelTracksCUDA",
    onGPU = True
)

phase2_tracker.toReplaceWith(pixelVerticesCUDA,_pixelVerticesCUDAPhase2.clone(
    pixelTrackSrc = "pixelTracksCUDA",
    onGPU = True,
    PtMin = 2.0
))

(pp_on_AA & ~phase2_tracker).toReplaceWith(pixelVerticesCUDA,_pixelVerticesCUDAHIonPhase1.clone(
    pixelTrackSrc = "pixelTracksCUDA",
    doSplitting = False,
    onGPU = True
))

# transfer the pixel vertices in SoA format to the CPU
from RecoVertex.PixelVertexFinding.pixelVerticesSoA_cfi import pixelVerticesSoA as _pixelVerticesSoA
gpu.toModify(pixelVerticesSoA,
    cuda = _pixelVerticesSoA.clone(
        src = cms.InputTag("pixelVerticesCUDA")
    )
)

## GPU vs CPU validation
# force CPU vertexing to use track SoA from CPU chain and not the converted one from GPU chain
from Configuration.ProcessModifiers.gpuValidationPixel_cff import gpuValidationPixel
(pixelNtupletFit & gpu & gpuValidationPixel).toModify(pixelVerticesSoA.cpu,
    pixelTrackSrc = "pixelTracksSoA@cpu"
)

(pixelNtupletFit & gpu).toReplaceWith(pixelVerticesTask, cms.Task(
    # build pixel vertices in SoA format on the GPU
    pixelVerticesCUDA,
    # transfer the pixel vertices in SoA format to the CPU and convert them to legacy format
    pixelVerticesTask.copy()
))

## pixel vertex reconstruction with Alpaka

# pixel vertex SoA producer with alpaka on the device
from RecoVertex.PixelVertexFinding.pixelVertexProducerAlpakaPhase1_cfi import pixelVertexProducerAlpakaPhase1 as _pixelVerticesAlpakaPhase1
from RecoVertex.PixelVertexFinding.pixelVertexProducerAlpakaPhase2_cfi import pixelVertexProducerAlpakaPhase2 as _pixelVerticesAlpakaPhase2
from RecoVertex.PixelVertexFinding.pixelVertexProducerAlpakaHIonPhase1_cfi import pixelVertexProducerAlpakaHIonPhase1 as _pixelVerticesAlpakaHIonPhase1
pixelVerticesAlpaka = _pixelVerticesAlpakaPhase1.clone()
phase2_tracker.toReplaceWith(pixelVerticesAlpaka,_pixelVerticesAlpakaPhase2.clone( maxVertices = 512))
(pp_on_AA & ~phase2_tracker).toReplaceWith(pixelVerticesAlpaka,_pixelVerticesAlpakaHIonPhase1.clone(doSplitting = False, maxVertices = 32))

from RecoVertex.PixelVertexFinding.pixelVertexFromSoAAlpaka_cfi import pixelVertexFromSoAAlpaka as _pixelVertexFromSoAAlpaka
# strip tracks
from RecoVertex.PixelVertexFinding.pixelVertexProducerAlpakaPhase1Strip_cfi import pixelVertexProducerAlpakaPhase1Strip as _pixelVertexProducerAlpakaPhase1Strip
from Configuration.ProcessModifiers.stripNtupletFit_cff import stripNtupletFit

(alpaka & stripNtupletFit & ~phase2_tracker).toReplaceWith(pixelVerticesAlpaka, _pixelVertexProducerAlpakaPhase1Strip.clone())

alpaka.toReplaceWith(pixelVertices, _pixelVertexFromSoAAlpaka.clone())

# pixel vertex SoA producer with alpaka on the cpu, for validation
pixelVerticesAlpakaSerial = makeSerialClone(pixelVerticesAlpaka,
    pixelTrackSrc = 'pixelTracksAlpakaSerial'
)

alpaka.toReplaceWith(pixelVerticesTask, cms.Task(
    # Build the pixel vertices in SoA format with alpaka on the device
    pixelVerticesAlpaka,
    # Build the pixel vertices in SoA format with alpaka on the cpu (if requested by the validation)
    pixelVerticesAlpakaSerial,
    # Convert the pixel vertices from SoA format (on the host) to the legacy format
    pixelVertices
))

# Tasks and Sequences
recopixelvertexingTask = cms.Task(
    pixelTracksTask,
    pixelVerticesTask
)
recopixelvertexing = cms.Sequence(recopixelvertexingTask)
