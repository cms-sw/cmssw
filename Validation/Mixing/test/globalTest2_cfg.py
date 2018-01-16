import FWCore.ParameterSet.Config as cms

process = cms.Process("PRODVAL2")
process.DaqMonitorROOTBackEnd = cms.Service("DaqMonitorROOTBackEnd")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/tmp/Cum_global.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.test = DQMStep1Module('GlobalTest',
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5),
    fileName = cms.string('GlobalHistos.root')
)

process.p = cms.Path(process.test)


