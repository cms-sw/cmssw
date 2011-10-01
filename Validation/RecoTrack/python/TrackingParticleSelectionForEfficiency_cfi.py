import FWCore.ParameterSet.Config as cms

TrackingParticleSelectionForEfficiency = cms.PSet(
    lipTP = cms.double(30.0),
    chargedOnlyTP = cms.bool(True),
    stableOnlyTP = cms.bool(False),
    pdgIdTP = cms.vint32(),
    signalOnlyTP = cms.bool(True),
    minRapidityTP = cms.double(-2.4),
    minHitTP = cms.int32(0),
    ptMinTP = cms.double(0.005),
    maxRapidityTP = cms.double(2.4),
    tipTP = cms.double(60)
)

