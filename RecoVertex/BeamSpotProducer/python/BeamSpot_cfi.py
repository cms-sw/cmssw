import FWCore.ParameterSet.Config as cms

import RecoVertex.BeamSpotProducer.Modifiers as mods

offlineBeamSpot = cms.EDProducer("BeamSpotProducer")

import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
_onlineBeamSpotProducer = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()
mods.offlineToOnlineBeamSpotSwap.toReplaceWith(offlineBeamSpot, _onlineBeamSpotProducer)
