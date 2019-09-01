import FWCore.ParameterSet.Config as cms

muonSimHitMatcherPSet = cms.PSet(
    simTrack = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("g4SimHits"),
    ),
    simVertex = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("g4SimHits"),
    ),
    gemSimHit = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("g4SimHits", "MuonGEMHits"),
        simMuOnly = cms.bool(True),
        discardEleHits = cms.bool(True),
    ),
    me0SimHit = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("g4SimHits", "MuonME0Hits"),
        simMuOnly = cms.bool(True),
        discardEleHits = cms.bool(True),
        minNHitsChamber = cms.int32(4),
    ),
    rpcSimHit = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("g4SimHits", "MuonRPCHits"),
        simMuOnly = cms.bool(True),
        discardEleHits = cms.bool(True),
    ),
    cscSimHit = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("g4SimHits", "MuonCSCHits"),
        simMuOnly = cms.bool(True),
        discardEleHits = cms.bool(True),
        minNHitsChamber = cms.int32(4),
    ),
    dtSimHit = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("g4SimHits", "MuonDTHits"),
        simMuOnly = cms.bool(True),
        discardEleHits = cms.bool(True),
        minNHitsChamber = cms.int32(4),
    )
)
