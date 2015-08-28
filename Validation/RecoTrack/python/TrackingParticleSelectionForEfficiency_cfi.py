import FWCore.ParameterSet.Config as cms

TrackingParticleSelectionForEfficiency = cms.PSet(
    lipTP = cms.double(30.0),
    chargedOnlyTP = cms.bool(True),
    stableOnlyTP = cms.bool(False),
    pdgIdTP = cms.vint32(),
    signalOnlyTP = cms.bool(False),
    intimeOnlyTP = cms.bool(True),
    minRapidityTP = cms.double(-2.5),
    minHitTP = cms.int32(0),
    ptMinTP = cms.double(0.005),
    maxRapidityTP = cms.double(2.5),
    tipTP = cms.double(60)
)

from Configuration.StandardSequences.Eras import eras
if eras.fastSim.isChosen():
    TrackingParticleSelectionForEfficiency.stableOnlyTP = True
