import FWCore.ParameterSet.Config as cms

cutsMuTPEffic = cms.EDFilter("TrackingParticleSelector",
    src = cms.InputTag("trackingParticles"),
    pdgId = cms.vint32(13, -13),
    tip = cms.double(3.5),
    minRapidity = cms.double(-2.4),
    lip = cms.double(30.0),
    ptMin = cms.double(0.9),
    maxRapidity = cms.double(2.4),
    minHit = cms.int32(0)
)



