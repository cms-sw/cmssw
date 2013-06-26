import FWCore.ParameterSet.Config as cms

cscRecHitValidation = cms.EDAnalyzer("CSCRecHitValidation",
    simHitsTag = cms.InputTag("mix","g4SimHitsMuonCSCHits"),
    outputFile = cms.string('CSCRecHitValidation.root'),
    recHitLabel = cms.InputTag("csc2DRecHits"),
    segmentLabel = cms.InputTag("cscSegments")
)


