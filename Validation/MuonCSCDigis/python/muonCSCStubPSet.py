import FWCore.ParameterSet.Config as cms

muonCSCStubPSet = cms.PSet(
    #csc CLCT, central BX 7
    cscCLCT = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("simCscTriggerPrimitiveDigis"),
        minBX = cms.int32(6),
        maxBX = cms.int32(8),
        minNHitsChamber = cms.int32(4),
    ),
    #csc ALCT, central BX 3 in CMSSW
    cscALCT = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("simCscTriggerPrimitiveDigis"),
        minBX = cms.int32(2),
        maxBX = cms.int32(4),
        minNHitsChamber = cms.int32(4),
    ),
    #csc LCT, central BX 8
    cscLCT = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("simCscTriggerPrimitiveDigis"),
        minBX = cms.int32(7),
        maxBX = cms.int32(9),
        minNHitsChamber = cms.int32(4),
        addGhosts = cms.bool(True)
    ),
    #csc LCT, central BX 8
    cscMPLCT = cms.PSet(
        verbose = cms.int32(0),
        inputTag = cms.InputTag("simCscTriggerPrimitiveDigis","MPCSORTED"),
        minBX = cms.int32(7),
        maxBX = cms.int32(9),
        minNHitsChamber = cms.int32(4),
    ),
)
