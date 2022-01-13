import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("PoolSource", 
                             fileNames = cms.untracked.vstring("file:stat_sender_second.root"),
                             secondaryFileNames = cms.untracked.vstring("file:stat_sender_first.root")
)

process.add_(cms.Service("StatisticsSenderService", debug = cms.untracked.bool(True)))