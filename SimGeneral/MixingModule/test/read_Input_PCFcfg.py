import FWCore.ParameterSet.Config as cms
 
process = cms.Process("PROInputA")
 
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring('file:/tmp/ebecheva/PCFwriter2.root')
)

 
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(1)
)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.InputAnalyzer = dict()
process.Analyzer = cms.EDAnalyzer("InputAnalyzer",
     dataStep2 = cms.bool(True),
     collPCF = cms.InputTag("CFWriter")
)
 
process.p = cms.Path(process.Analyzer)
