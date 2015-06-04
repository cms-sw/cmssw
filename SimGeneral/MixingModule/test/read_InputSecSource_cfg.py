import FWCore.ParameterSet.Config as cms
 
process = cms.Process("PROInputAnal")
 
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        Analyzer = cms.untracked.uint32(12345)
    )
)

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:/data/becheva/MixingModule/dataFiles/relval/02C5A172-8203-DE11-86D7-001617C3B5D8TTBar.root')
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(1)
)

process.Analyzer = cms.EDAnalyzer("SecSourceAnalyzer",
     dataStep2 = cms.bool(False),
     collSimTrack = cms.InputTag("g4SimHits"),
	
     input = cms.SecSource("EmbeddedRootSource",
        type = cms.string('fixed'),
        nbPileupEvents = cms.PSet(
        averageNumber = cms.double(1.0)
        ),
        fileNames = cms.untracked.vstring('file:/data/becheva/MixingModule/dataFiles/relval/02C5A172-8203-DE11-86D7-001617C3B5D8TTBar.root')
     )
)
 
process.p = cms.Path(process.Analyzer)
