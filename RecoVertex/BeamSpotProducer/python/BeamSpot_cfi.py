import FWCore.ParameterSet.Config as cms

import RecoVertex.BeamSpotProducer.Modifiers as mods

offlineBeamSpot = cms.EDProducer("BeamSpotProducer")

def _loadOnlineBeamSpotESProducer(process):
    import RecoVertex.BeamSpotProducer.onlineBeamSpotESProducer_cfi as _mod
    process.BeamSpotESProducer = _mod.onlineBeamSpotESProducer.clone(
        timeThreshold = 999999 # for express allow >48h old payloads for replays. DO NOT CHANGE
    )

import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
_onlineBeamSpotProducer = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()
mods.offlineToOnlineBeamSpotSwap.toReplaceWith(offlineBeamSpot, _onlineBeamSpotProducer)

applyOnlineBSESProducer = mods.offlineToOnlineBeamSpotSwap.makeProcessModifier(_loadOnlineBeamSpotESProducer)
