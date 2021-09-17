import FWCore.ParameterSet.Config as cms

muonCSCDigiPSet = cms.PSet(
    #csc comparator digi, central BX 7
    cscComparatorDigi = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("simMuonCSCDigis", "MuonCSCComparatorDigi"),
        minBX = cms.int32(4),
        maxBX = cms.int32(10),
        matchDeltaStrip = cms.int32(2),
        minNHitsChamber = cms.int32(4),
    ),
    cscStripDigi = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("simMuonCSCDigis", "MuonCSCStripDigi"),
        minBX = cms.int32(4),
        maxBX = cms.int32(10),
        matchDeltaStrip = cms.int32(2),
        minNHitsChamber = cms.int32(4),
    ),
    #csc wire digi, central BX 8
    cscWireDigi = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("simMuonCSCDigis", "MuonCSCWireDigi"),
        minBX = cms.int32(5),
        maxBX = cms.int32(11),
        matchDeltaWG = cms.int32(2),
        minNHitsChamber = cms.int32(4),
    ),
)
