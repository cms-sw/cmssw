import FWCore.ParameterSet.Config as cms

import PhysicsTools.RecoAlgos.recoTrackSelector_cfi
cutsRecoCTFTracksP5 = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone(
    src = "ctfWithMaterialTracksP5",
    maxChi2 = cms.double(10000.0),
    tip = cms.double(120.0),
    lip = cms.double(300.0),
    ptMin = cms.double(0.1),
    maxRapidity = cms.double(5.0),
    quality = cms.vstring(''),
    algorithm = cms.vstring(),
    minHit = cms.int32(3),
    min3DHit = cms.int32(0),
    beamSpot = cms.InputTag("offlineBeamSpot")
    )


