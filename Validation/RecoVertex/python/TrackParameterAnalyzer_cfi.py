import FWCore.ParameterSet.Config as cms

simpleTrackAnalysis = cms.EDAnalyzer("TrackParameterAnalyzer",
    simG4 = cms.InputTag("g4SimHits"),
    outputFile = cms.untracked.string('simpleTrackParameterAnalyzer.root'),
    recoTrackProducer = cms.untracked.string('generalTracks'),
    verbose = cms.untracked.bool(True)
)



