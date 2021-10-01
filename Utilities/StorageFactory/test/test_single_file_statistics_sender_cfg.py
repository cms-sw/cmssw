import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:stat_sender_first.root"))

process.add_(cms.Service("StatisticsSenderService", debug = cms.untracked.bool(True)))

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.cerr.INFO.limit = 1000