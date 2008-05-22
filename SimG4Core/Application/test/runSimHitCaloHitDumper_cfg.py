import FWCore.ParameterSet.Config as cms

process = cms.Process("SimCaloHitDump")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:myfile.root')
)

process.prod = cms.EDFilter("SimHitCaloHitDumper")

process.p1 = cms.Path(process.prod)


