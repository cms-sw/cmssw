import FWCore.ParameterSet.Config as cms

process = cms.Process("ContainmentCorrectionAnalysis")
process.load("Configuration.StandardSequences.GeometryExtended_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:245E3E90-0741-DD11-BB73-000423D99660.root')
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('contCorrAnalyzer.root')
)

process.contCorrAnalyzer = cms.EDAnalyzer("ContainmentCorrectionAnalyzer",
    BarrelSuperClusterCollection = cms.InputTag("correctedHybridSuperClusters"),
    reducedEndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
    simVertexCollection = cms.InputTag("g4SimHits"),
    simTrackCollection = cms.InputTag("g4SimHits"),
    EndcapSuperClusterCollection = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"),
    reducedBarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB")
)

process.p = cms.Path(process.contCorrAnalyzer)

