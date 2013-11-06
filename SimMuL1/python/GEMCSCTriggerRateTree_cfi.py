import FWCore.ParameterSet.Config as cms

GEMCSCTriggerRateTree = cms.EDFilter("GEMCSCTriggerRateTree",
    doME1a = cms.untracked.bool(True),
    defaultME1a = cms.untracked.bool(False),
    minBxALCT = cms.untracked.int32(5),
    maxBxALCT = cms.untracked.int32(7),
    minBxCLCT = cms.untracked.int32(5),
    maxBxCLCT = cms.untracked.int32(7),
    minBxLCT = cms.untracked.int32(5),
    maxBxLCT = cms.untracked.int32(7),
    minBxMPLCT = cms.untracked.int32(5),
    maxBxMPLCT = cms.untracked.int32(7),
    sectorProcessor = cms.untracked.PSet(),
    strips = cms.untracked.PSet()
)
