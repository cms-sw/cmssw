import FWCore.ParameterSet.Config as cms
from SimGeneral.MixingModule.trackingTruthProducer_cfi import trackingParticles

trackingParticlesSelection = cms.PSet(
    lipTP = cms.double(1000),
    chargedOnlyTP = cms.bool(True),
    stableOnlyTP = cms.bool(False),
    pdgIdTP = cms.vint32(),
    signalOnlyTP = cms.bool(False),
    intimeOnlyTP = cms.bool(False),
    minRapidityTP = cms.double(-5.0),
    minHitTP = cms.int32(0),
    ptMinTP = cms.double(0.1),
    maxRapidityTP = cms.double(5.0),
    tipTP = cms.double(1000)
)

trackingParticles.select = cms.PSet(trackingParticlesSelection)
