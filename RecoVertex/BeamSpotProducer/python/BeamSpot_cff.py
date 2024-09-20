import FWCore.ParameterSet.Config as cms

from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *
offlineBeamSpotTask = cms.Task(offlineBeamSpot)

from RecoVertex.BeamSpotProducer.beamSpotDeviceProducer_cfi import beamSpotDeviceProducer as _beamSpotDeviceProducer
offlineBeamSpotDevice = _beamSpotDeviceProducer.clone(src = cms.InputTag('offlineBeamSpot'))

from Configuration.ProcessModifiers.alpaka_cff import alpaka
_offlineBeamSpotTask_alpaka = offlineBeamSpotTask.copy()
_offlineBeamSpotTask_alpaka.add(offlineBeamSpotDevice)
alpaka.toReplaceWith(offlineBeamSpotTask, _offlineBeamSpotTask_alpaka)
