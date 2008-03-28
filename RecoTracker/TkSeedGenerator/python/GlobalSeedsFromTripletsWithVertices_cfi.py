import FWCore.ParameterSet.Config as cms

globalSeedsFromTripletsWithVertices = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    #include "RecoTracker/PixelStubs/data/SeedComparitorWithPixelStubs.cfi"
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        SeedingLayers = cms.string('PixelLayerTriplets'),
        GeneratorPSet = cms.PSet(
            useBending = cms.bool(True),
            useFixedPreFiltering = cms.bool(False),
            ComponentName = cms.string('PixelTripletHLTGenerator'),
            extraHitRPhitolerance = cms.double(0.06),
            useMultScattering = cms.bool(True),
            phiPreFiltering = cms.double(0.3),
            extraHitRZtolerance = cms.double(0.06)
        )
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalRegionProducer'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originHalfLength = cms.double(15.9),
            originRadius = cms.double(0.2),
            originYPos = cms.double(0.0),
            ptMin = cms.double(0.9),
            originXPos = cms.double(0.0),
            originZPos = cms.double(0.0)
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)


