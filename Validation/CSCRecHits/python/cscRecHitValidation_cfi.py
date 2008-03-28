import FWCore.ParameterSet.Config as cms

cscRecHitValidation = cms.EDFilter("CSCRecHitValidation",
    outputFile = cms.string('CSCRecHitValidation.root'),
    recHitLabel = cms.InputTag("csc2DRecHits"),
    segmentLabel = cms.InputTag("cscSegments")
)


