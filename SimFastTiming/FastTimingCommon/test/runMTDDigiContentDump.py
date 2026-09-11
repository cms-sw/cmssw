import FWCore.ParameterSet.Config as cms

process = cms.Process("MTDDigiContentDump")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:step2.root'
    )
)

process.MTDDigiContentDump = cms.EDAnalyzer('MTDDigiContentDump')


process.p = cms.Path(process.MTDDigiContentDump)
