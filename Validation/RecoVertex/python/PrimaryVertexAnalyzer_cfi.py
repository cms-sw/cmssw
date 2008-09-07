import FWCore.ParameterSet.Config as cms

simpleVertexAnalysis = cms.EDAnalyzer("PrimaryVertexAnalyzer",
    simG4 = cms.InputTag("g4SimHits"),
    outputFile = cms.untracked.string('simpleVertexAnalyzer.root'),
    vtxSample = cms.untracked.string('offlinePrimaryVertices'),
    verbose = cms.untracked.bool(True),
    recoTrackProducer = cms.untracked.string('generalTracks')
)


