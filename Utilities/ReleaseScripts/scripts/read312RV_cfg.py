
import FWCore.ParameterSet.Config as cms

process = cms.Process("READTEST")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32(100) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
                 '/store/data/Run2010A/MinimumBias/RAW/v1/000/137/289/B419A444-BD73-DF11-A22C-0030487CD812.root')
)

process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )

# output module
#

process.out = cms.OutputModule("PoolOutputModule",
                               fastCloning=cms.untracked.bool(False),
                               fileName = cms.untracked.string('merge.root')
)

process.outpath = cms.EndPath(process.out)

