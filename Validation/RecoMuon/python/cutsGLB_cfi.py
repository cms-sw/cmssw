import FWCore.ParameterSet.Config as cms

cutsGLB = cms.EDFilter("RecoTrackSelector",
    src = cms.InputTag("globalMuons"),
    tip = cms.double(3.5),
    minRapidity = cms.double(-2.5),
    lip = cms.double(30.0),
    ptMin = cms.double(0.8),
    maxRapidity = cms.double(2.5),
    minHit = cms.int32(8)
)



