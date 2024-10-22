import FWCore.ParameterSet.Config as cms

process = cms.Process("tPtVDump")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:step3.root')
)

process.load("SimGeneral.CaloAnalysis.mtdTruthDumper_cfi")

process.p1 = cms.Path(process.mtdTruthDumper)
