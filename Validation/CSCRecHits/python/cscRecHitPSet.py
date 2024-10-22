import FWCore.ParameterSet.Config as cms

cscRecHitPSet = cms.PSet(
    cscRecHit = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("csc2DRecHits"),
        minBX = cms.int32(-1),
        maxBX = cms.int32(1),
    ),
    cscSegment = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("cscSegments"),
        minBX = cms.int32(-1),
        maxBX = cms.int32(1),
    )
)
