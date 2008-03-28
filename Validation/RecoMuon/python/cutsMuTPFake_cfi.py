import FWCore.ParameterSet.Config as cms

cutsMuTPFake = cms.EDFilter("TrackingParticleSelector",
    src = cms.InputTag("trackingParticles"),
    pdgId = cms.vint32(13, -13),
    tip = cms.double(120.0),
    minRapidity = cms.double(-2.6),
    lip = cms.double(250.0),
    ptMin = cms.double(0.7),
    maxRapidity = cms.double(2.6),
    minHit = cms.int32(0)
)


