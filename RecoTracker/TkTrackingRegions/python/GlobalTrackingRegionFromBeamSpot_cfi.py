import FWCore.ParameterSet.Config as cms

RegionPsetFomBeamSpotBlock = cms.PSet(
    RegionPSet = cms.PSet(
        precise = cms.bool(True),
        nSigmaZ = cms.double(3.0),
        originRadius = cms.double(0.2),
        ptMin = cms.double(0.9),
        beamSpot = cms.InputTag("offlineBeamSpot")
    )
)

