import FWCore.ParameterSet.Config as cms

process = cms.Process("MTDDigiDump")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:step2.root'
    )
)

process.MTDDigiDump = cms.EDAnalyzer('MTDDigiDump')


process.p = cms.Path(process.MTDDigiDump)
