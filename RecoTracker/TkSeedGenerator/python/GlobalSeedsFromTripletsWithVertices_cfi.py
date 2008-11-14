import FWCore.ParameterSet.Config as cms

from RecoTracker.TkTrackingRegions.GlobalTrackingRegion_cfi import *
from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *
globalSeedsFromTripletsWithVertices = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    #include "RecoTracker/PixelStubs/data/SeedComparitorWithPixelStubs.cfi"
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        SeedingLayers = cms.string('PixelLayerTriplets'),
        GeneratorPSet = cms.PSet(
            PixelTripletHLTGenerator
        )
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        RegionPSetBlock,
        ComponentName = cms.string('GlobalRegionProducer')
    ),
    propagator = cms.string('PropagatorWithMaterial'),
# The fast-helix fit works well, except for large impact parameter pixel pair seeding.                                                     
    UseFastHelix = cms.bool(True),
# Following parameter not relevant for UseFastHelix = False.                                                                                        
    SeedMomentumForBOFF = cms.double(5.0), 
    TTRHBuilder = cms.string('WithTrackAngle')
)


