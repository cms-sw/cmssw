
import FWCore.ParameterSet.Config as cms

process = cms.Process("READTEST")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32(100) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
                 '/store/data/Run2010A/MinimumBias/RAW/v1/000/140/386/02FD0631-A092-DF11-9E12-001D09F290CE.root'
    )
)

process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )

# output module
#

process.out = cms.OutputModule("PoolOutputModule",
                               fastCloning=cms.untracked.bool(False),
                               fileName = cms.untracked.string('merge.root')
)

process.outpath = cms.EndPath(process.out)

