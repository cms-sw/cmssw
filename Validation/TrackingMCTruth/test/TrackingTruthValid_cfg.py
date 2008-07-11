import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Services_cff")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")

process.load("Validation.TrackingMCTruth.trackingTruthValidation_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/6/22/RelVal-RelValSingleMuPt10-1214048167-IDEAL_V2-2nd/0004/0AE2B3E3-0141-DD11-846F-000423D98BC4.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.mix*process.trackingParticles*process.trackingTruthValid)


