import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import *

from RecoTracker.PixelTrackFitting.PixelTracks_cff import *
from RecoVertex.PixelVertexFinding.PixelVertices_cff import *

# HIon Modifiers
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
# Phase 2 Tracker Modifier
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker

# legacy pixel vertex reconsruction using the divisive vertex finder
pixelVerticesTask = cms.Task(
    pixelVertices
)

############################################################
## Pixel vertex reconstruction with Alpaka

# pixel vertex SoA producer with alpaka on the device
from RecoVertex.PixelVertexFinding.pixelVertexProducerAlpakaPhase1_cfi import pixelVertexProducerAlpakaPhase1 as _pixelVerticesAlpakaPhase1
from RecoVertex.PixelVertexFinding.pixelVertexProducerAlpakaPhase2_cfi import pixelVertexProducerAlpakaPhase2 as _pixelVerticesAlpakaPhase2
from RecoVertex.PixelVertexFinding.pixelVertexProducerAlpakaHIonPhase1_cfi import pixelVertexProducerAlpakaHIonPhase1 as _pixelVerticesAlpakaHIonPhase1
pixelVerticesAlpaka = _pixelVerticesAlpakaPhase1.clone()
phase2_tracker.toReplaceWith(pixelVerticesAlpaka,_pixelVerticesAlpakaPhase2.clone( maxVertices = 512))
(pp_on_AA & ~phase2_tracker).toReplaceWith(pixelVerticesAlpaka,_pixelVerticesAlpakaHIonPhase1.clone(doSplitting = False, maxVertices = 32))

from RecoVertex.PixelVertexFinding.pixelVertexFromSoAAlpaka_cfi import pixelVertexFromSoAAlpaka as _pixelVertexFromSoAAlpaka
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
