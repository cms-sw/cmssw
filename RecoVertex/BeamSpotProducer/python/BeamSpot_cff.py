import FWCore.ParameterSet.Config as cms

from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *
from RecoVertex.BeamSpotProducer.offlineBeamSpotToCUDA_cfi import offlineBeamSpotToCUDA
from RecoVertex.BeamSpotProducer.beamSpotDeviceProducer_cfi import beamSpotDeviceProducer as _beamSpotDeviceProducer

offlineBeamSpotTask = cms.Task(offlineBeamSpot)

from Configuration.ProcessModifiers.gpu_cff import gpu
_offlineBeamSpotTask_gpu = offlineBeamSpotTask.copy()
_offlineBeamSpotTask_gpu.add(offlineBeamSpotToCUDA)
gpu.toReplaceWith(offlineBeamSpotTask, _offlineBeamSpotTask_gpu)

from Configuration.ProcessModifiers.alpaka_cff import alpaka
_offlineBeamSpotTask_alpaka = offlineBeamSpotTask.copy()
offlineBeamSpotDevice = _beamSpotDeviceProducer.clone(src = cms.InputTag('offlineBeamSpot'))
_offlineBeamSpotTask_alpaka.add(offlineBeamSpotDevice)
alpaka.toReplaceWith(offlineBeamSpotTask, _offlineBeamSpotTask_alpaka)
