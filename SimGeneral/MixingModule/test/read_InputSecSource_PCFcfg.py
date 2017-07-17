import FWCore.ParameterSet.Config as cms
 
process = cms.Process("PROInputAnal")
 
#process.source = cms.Source("EmptySource")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        Analyzer = cms.untracked.uint32(12345)
    )
)

process.source = cms.Source("PoolSource",
  #fileNames = cms.untracked.vstring('file:/data/becheva/MixingModule/dataFiles/relval/02C5A172-8203-DE11-86D7-001617C3B5D8TTBar.root')
  fileNames = cms.untracked.vstring('file:/tmp/ebecheva/PCFwriter2.root')
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(1)
)

process.Analyzer = cms.EDAnalyzer("SecSourceAnalyzer",
     dataStep2 = cms.bool(True),
     collPCF = cms.InputTag("CFWriter"),
	
     input = cms.SecSource("EmbeddedRootSource",
        type = cms.string('fixed'),
        nbPileupEvents = cms.PSet(
        averageNumber = cms.double(1.0)
        ),
        fileNames = cms.untracked.vstring('file:/tmp/ebecheva/PCFwriter2.root')
     )
)
 
process.p = cms.Path(process.Analyzer)
