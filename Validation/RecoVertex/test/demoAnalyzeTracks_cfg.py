import FWCore.ParameterSet.Config as cms

process = cms.Process("AnalyzeTracks")
#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10))
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:pv_reco.root')
)

process.simpleTrackParameterAnalysis = cms.EDAnalyzer("TrackParameterAnalyzer",
    simG4 = cms.InputTag("g4SimHits"),
    outputFile = cms.untracked.string('simpleTrackAnalyzer.root'),
    verbose = cms.untracked.bool(True),
    vtxSample = cms.untracked.string('offlinePrimaryVertices'),
    recoTrackProducer = cms.untracked.string('generalTracks')
)

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.Tracer = cms.Service("Tracer",
    indention = cms.untracked.string('$$')
)

process.p = cms.Path(process.simpleTrackParameterAnalysis*process.dump)


