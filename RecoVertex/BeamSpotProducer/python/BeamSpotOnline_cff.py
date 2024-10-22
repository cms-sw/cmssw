import FWCore.ParameterSet.Config as cms

import RecoVertex.BeamSpotProducer.onlineBeamSpotESProducer_cfi as _mod
BeamSpotESProducer = _mod.onlineBeamSpotESProducer.clone()

from RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi import *
onlineBeamSpot = cms.Sequence( onlineBeamSpotProducer )

