import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Services_cff")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")

process.load("Validation.TrackingMCTruth.trackingTruthValidation_cfi")
process.trackingTruthValid.outputFile = "trackingtruthhisto.root"

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/38E34C97-E8DD-DD11-8327-000423D94534.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.mix*process.trackingParticles*process.trackingTruthValid)


