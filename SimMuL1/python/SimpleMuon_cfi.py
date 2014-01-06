import FWCore.ParameterSet.Config as cms

from GEMCode.GEMValidation.simTrackMatching_cfi import SimTrackMatching

SimpleMuon = cms.EDAnalyzer('SimpleMuon',
    debugALLEVENT = cms.untracked.int32(0),
    debugINHISTOS = cms.untracked.int32(0),
    debugALCT     = cms.untracked.int32(0),
    debugCLCT     = cms.untracked.int32(0),
    debugLCT      = cms.untracked.int32(0),
    debugMPLCT    = cms.untracked.int32(0),
    debugTFTRACK  = cms.untracked.int32(0),
    debugTFCAND   = cms.untracked.int32(0),
    debugGMTCAND  = cms.untracked.int32(0),
    debugL1EXTRA  = cms.untracked.int32(0),
    strips = cms.PSet()                 
)    
