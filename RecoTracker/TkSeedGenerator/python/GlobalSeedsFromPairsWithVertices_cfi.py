import FWCore.ParameterSet.Config as cms

globalSeedsFromPairsWithVertices = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    #include "RecoTracker/PixelStubs/data/SeedComparitorWithPixelStubs.cfi"
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.string('MixedLayerPairs')
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalTrackingRegionWithVerticesProducer'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            useFixedError = cms.bool(True),
            originRadius = cms.double(0.2),
            sigmaZVertex = cms.double(3.0),
            fixedError = cms.double(0.2),
            VertexCollection = cms.string('pixelVertices'),
            ptMin = cms.double(0.9),
            useFoundVertices = cms.bool(True),
            nSigmaZ = cms.double(3.0)
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)


