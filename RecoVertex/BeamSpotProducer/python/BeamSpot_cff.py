import FWCore.ParameterSet.Config as cms

from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *
from RecoVertex.BeamSpotProducer.beamSpotToCUDA_cfi import beamSpotToCUDA as _beamSpotToCUDA

offlineBeamSpotTask = cms.Task(offlineBeamSpot)

from Configuration.ProcessModifiers.gpu_cff import gpu
offlineBeamSpotCUDA = _beamSpotToCUDA.clone()
_offlineBeamSpotTask_gpu = offlineBeamSpotTask.copy()
_offlineBeamSpotTask_gpu.add(offlineBeamSpotCUDA)
gpu.toReplaceWith(offlineBeamSpotTask, _offlineBeamSpotTask_gpu)
