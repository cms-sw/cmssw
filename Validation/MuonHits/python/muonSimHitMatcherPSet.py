import FWCore.ParameterSet.Config as cms

muonSimHitMatcherPSet = cms.PSet(
    simTrack = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("g4SimHits"),
        minPt = cms.double(5.0),
        minEta = cms.double(0),
        maxEta = cms.double(2.8),
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

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(muonSimHitMatcherPSet,
                 simTrack = dict(inputTag = "fastSimProducer"),
                 simVertex = dict(inputTag = "fastSimProducer"),
                 gemSimHit = dict(inputTag = "MuonSimHits:MuonGEMHits"),
                 me0SimHit = dict(inputTag = "MuonSimHits:MuonME0Hits"),
                 rpcSimHit = dict(inputTag = "MuonSimHits:MuonRPCHits"),
                 cscSimHit = dict(inputTag = "MuonSimHits:MuonCSCHits"),
                 dtSimHit = dict(inputTag = "MuonSimHits:MuonDTHits"),
)
