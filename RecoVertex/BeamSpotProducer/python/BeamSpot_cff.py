import FWCore.ParameterSet.Config as cms

from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *
from RecoVertex.BeamSpotProducer.offlineBeamSpotCUDA_cfi import offlineBeamSpotCUDA

offlineBeamSpotTask = cms.Task(offlineBeamSpot)

from Configuration.ProcessModifiers.gpu_cff import gpu
_offlineBeamSpotTask_gpu = offlineBeamSpotTask.copy()
_offlineBeamSpotTask_gpu.add(offlineBeamSpotCUDA)
gpu.toReplaceWith(offlineBeamSpotTask, _offlineBeamSpotTask_gpu)
