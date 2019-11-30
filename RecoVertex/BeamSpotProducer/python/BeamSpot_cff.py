import FWCore.ParameterSet.Config as cms

from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *
from RecoVertex.BeamSpotProducer.beamSpotToCUDA_cfi import beamSpotToCUDA as _beamSpotToCUDA

offlineBeamSpotCUDA = _beamSpotToCUDA.clone()

offlineBeamSpotTask = cms.Task(
    offlineBeamSpot,
    offlineBeamSpotCUDA
)
