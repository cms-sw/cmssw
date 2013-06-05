import FWCore.ParameterSet.Config as cms
from SimGeneral.MixingModule.trackingTruthProducer_cfi import trackingParticles

trackingParticles.select = cms.PSet(
    lipTP = cms.double(1000),
    chargedOnlyTP = cms.bool(True),
    pdgIdTP = cms.vint32(),
    signalOnlyTP = cms.bool(True),
    minRapidityTP = cms.double(-2.6),
    minHitTP = cms.int32(3),
    ptMinTP = cms.double(0.2),
    maxRapidityTP = cms.double(2.6),
    tipTP = cms.double(1000)
)
