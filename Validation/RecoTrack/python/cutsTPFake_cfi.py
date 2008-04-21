import FWCore.ParameterSet.Config as cms

cutsTPFake = cms.EDFilter("TrackingParticleSelector",
    src = cms.InputTag("mergedtruth","MergedTrackTruth"),
    chargedOnly = cms.bool(True),
    pdgId = cms.vint32(),
    tip = cms.double(120.0),
    signalOnly = cms.bool(False),
    minRapidity = cms.double(-3.0),
    lip = cms.double(250.0),
    ptMin = cms.double(0.1),
    maxRapidity = cms.double(3.0),
    minHit = cms.int32(0)
)


