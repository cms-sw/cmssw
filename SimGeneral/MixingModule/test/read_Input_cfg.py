import FWCore.ParameterSet.Config as cms
 
process = cms.Process("PROInputA")
 
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:/data/becheva/MixingModule/dataFiles/relval/02C5A172-8203-DE11-86D7-001617C3B5D8TTBar.root')
)


process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(1)
)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.InputAnalyzer = dict()
process.Analyzer = cms.EDAnalyzer("InputAnalyzer",
     dataStep2 = cms.bool(False),
     collSimTrack = cms.InputTag("g4SimHits")
)
 
process.p = cms.Path(process.Analyzer)
