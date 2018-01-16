import FWCore.ParameterSet.Config as cms

cscRecHitValidation = DQMStep1Module('CSCRecHitValidation',
    simHitsTag = cms.InputTag("mix","g4SimHitsMuonCSCHits"),
    outputFile = cms.string('CSCRecHitValidation.root'),
    recHitLabel = cms.InputTag("csc2DRecHits"),
    segmentLabel = cms.InputTag("cscSegments")
)


