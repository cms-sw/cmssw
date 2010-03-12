import FWCore.ParameterSet.Config as cms

pixelVZeros = cms.EDProducer("VZeroProducer",
    minImpactPositiveDaughter = cms.double(0.0),
    minImpactNegativeDaughter = cms.double(0.0),
    #
    maxDca = cms.double(0.2),
    #
    minCrossingRadius = cms.double(0.2),
    maxCrossingRadius = cms.double(5.0),
    #
    maxImpactMother = cms.double(0.2)
)


