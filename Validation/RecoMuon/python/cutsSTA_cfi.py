import FWCore.ParameterSet.Config as cms

cutsSTA = cms.EDFilter("RecoTrackSelector",
    src = cms.InputTag("standAloneMuons","UpdatedAtVtx"),
    tip = cms.double(999.0),
    minRapidity = cms.double(-2.5),
    lip = cms.double(999.0),
    ptMin = cms.double(0.8),
    maxRapidity = cms.double(2.5),
    minHit = cms.int32(1)
)



