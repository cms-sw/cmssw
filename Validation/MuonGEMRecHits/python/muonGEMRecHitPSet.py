import FWCore.ParameterSet.Config as cms

gemRecHit = cms.PSet(
    verbose = cms.int32(0),
    inputTag = cms.InputTag("gemRecHits"),
    minBX = cms.int32(-1),
    maxBX = cms.int32(1),
)
