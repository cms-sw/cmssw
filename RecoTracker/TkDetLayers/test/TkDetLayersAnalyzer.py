import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")

process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32(1) )

process.load("FWCore.MessageLogger.MessageLogger_cfi")


process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'DEBUG'
# 
process.MessageLogger.debugModules = ['analyzer']
#process.MessageLogger.cerr.DEBUG = cms.untracked.PSet(
#    threshold = cms.untracked.string('DEBUG'),
#    default          = cms.untracked.PSet( limit = cms.untracked.int32(0)  )
#    TkDetLayers = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
#    )

#cat debug | grep -v MSG | grep -v "Run:" | grep -v analyzer > debug.readable

process.source = cms.Source("EmptySource")

process.analyzer = cms.EDAnalyzer("TkDetLayersAnalyzer")
process.p1 = cms.Path(process.analyzer)


