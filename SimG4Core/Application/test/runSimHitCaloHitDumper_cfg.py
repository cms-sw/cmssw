import FWCore.ParameterSet.Config as cms

process = cms.Process("SimCaloHitDump")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:step1.root')
)

process.load("SimG4Core.Application.simHitCaloHitDumper_cfi")

process.p1 = cms.Path(process.simHitCaloHitDumper)
