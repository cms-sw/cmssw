import FWCore.ParameterSet.Config as cms

pixelVZeros = cms.EDProducer("VZeroProducer",
    #
    trackCollection  = cms.InputTag("allTracks"),
    vertexCollection = cms.InputTag("pixel3Vertices"),
    #
    minImpactPositiveDaughter = cms.double(0.2),
    minImpactNegativeDaughter = cms.double(0.2),
    #
    maxDca = cms.double(0.2),
    #
    minCrossingRadius = cms.double(0.2),
    maxCrossingRadius = cms.double(5.0),
    #
    maxImpactMother = cms.double(0.2)
)

