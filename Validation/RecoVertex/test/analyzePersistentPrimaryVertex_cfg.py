import FWCore.ParameterSet.Config as cms

process = cms.Process("AnalyzePersistentVertices")
#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")




process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:pv_reco.root')
)

process.simplePersistentVertexAnalysis = cms.EDAnalyzer("PrimaryVertexAnalyzer",
    simG4 = cms.InputTag("g4SimHits"),
    outputFile = cms.untracked.string('simplePersistentVertexAnalyzer.root'),
    verbose = cms.untracked.bool(True),
    vtxSample = cms.untracked.string('offlinePrimaryVertices'),
    recoTrackProducer = cms.untracked.string('generalTracks')
)

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.Tracer = cms.Service("Tracer",
    indention = cms.untracked.string('$$')
)

process.p = cms.Path(process.simplePersistentVertexAnalysis)


