import FWCore.ParameterSet.Config as cms

RegionPsetFomBeamSpotBlock = cms.PSet(
    RegionPSet = cms.PSet(
        precise = cms.bool(True),
        nSigmaZ = cms.double(4.0),
        originRadius = cms.double(0.2),
        ptMin = cms.double(0.9),
        beamSpot = cms.InputTag("offlineBeamSpot")
    )
)

RegionPsetFomBeamSpotBlockFixedZ = cms.PSet(
    RegionPSet = cms.PSet(
        precise = cms.bool(True),
        originHalfLength = cms.double(21.2),
        originRadius = cms.double(0.2),
        ptMin = cms.double(0.9),
        beamSpot = cms.InputTag("offlineBeamSpot")
    )
)

