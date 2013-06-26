import FWCore.ParameterSet.Config as cms

process = cms.Process("PRODVAL2")
process.load("DQM.SiStripCommon.DaqMonitorROOTBackEnd_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/tmp/Cum_xxx.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.test = cms.EDAnalyzer("TestSuite",
    maxBunch = cms.int32(34567),
    BunchNr = cms.int32(12345),
    minBunch = cms.int32(23456),
    fileName = cms.string('histos.root')
)

#process.DaqMonitorROOTBackEnd = cms.Service("DaqMonitorROOTBackEnd")

process.p = cms.Path(process.test)


