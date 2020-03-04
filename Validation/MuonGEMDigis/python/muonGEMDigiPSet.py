import FWCore.ParameterSet.Config as cms

muonGEMDigiPSet = cms.PSet(
    gemStripDigi = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("simMuonGEMDigis"),
        minBX = cms.int32(-1),
        maxBX = cms.int32(1),
        matchDeltaStrip = cms.int32(1),
    ),
    gemUnpackedStripDigi = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("muonGEMDigis"),
        minBX = cms.int32(-1),
        maxBX = cms.int32(1),
        matchDeltaStrip = cms.int32(1),
    ),
    gemPadDigi = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("simMuonGEMPadDigis"),
        minBX = cms.int32(-1),
        maxBX = cms.int32(1),
     ),
    gemPadCluster = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("simMuonGEMPadDigiClusters"),
        minBX = cms.int32(-1),
        maxBX = cms.int32(1),
     ),
    gemCoPadDigi = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("simCscTriggerPrimitiveDigis"),
        minBX = cms.int32(-1),
        maxBX = cms.int32(1),
    ),
)
