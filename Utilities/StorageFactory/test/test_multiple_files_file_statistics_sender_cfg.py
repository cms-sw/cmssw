import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:stat_sender_b.root", "file:stat_sender_c.root", "file:stat_sender_d.root", "file:stat_sender_e.root"),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
inputCommands = cms.untracked.vstring('drop *_*_beginRun_*', 'drop *_*_endRun_*', 'drop *_*_beginLumi_*', 'drop *_*_endLumi_*')
)

process.add_(cms.Service("StatisticsSenderService", debug = cms.untracked.bool(True)))