import FWCore.ParameterSet.Config as cms

process = cms.Process("SECOND")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:stat_sender_first.root"))

process.o = cms.OutputModule("PoolOutputModule", fileName = cms.untracked.string("stat_sender_second.root"), outputCommands = cms.untracked.vstring("drop *"))

process.ep = cms.EndPath(process.o)

