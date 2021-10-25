import FWCore.ParameterSet.Config as cms

muonGEMDigiPSet = cms.PSet(
    gemSimLink = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("simMuonGEMDigis","GEM"),
        simMuOnly = cms.bool(True),
        discardEleHits = cms.bool(True),
    ),
    gemStripDigi = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("simMuonGEMDigis"),
        minBX = cms.int32(-1),
        maxBX = cms.int32(1),
        matchDeltaStrip = cms.int32(1),
        matchToSimLink = cms.bool(True)
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
        minBX = cms.int32(0),
        maxBX = cms.int32(0),
    ),
)

from Configuration.Eras.Modifier_phase2_GEM_cff import phase2_GEM
phase2_GEM.toModify(muonGEMDigiPSet.gemUnpackedStripDigi, inputTag = "simMuonGEMDigis")
