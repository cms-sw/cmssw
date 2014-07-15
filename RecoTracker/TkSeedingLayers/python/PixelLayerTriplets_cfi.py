import FWCore.ParameterSet.Config as cms

from RecoTracker.TkSeedingLayers.seedingLayersEDProducer_cfi import *

PixelLayerTriplets = seedingLayersEDProducer.clone()
PixelLayerTriplets.layerList = cms.vstring('BPix1+BPix2+BPix3', 
    'BPix1+BPix2+FPix1_pos', 
    'BPix1+BPix2+FPix1_neg', 
    'BPix1+FPix1_pos+FPix2_pos', 
    'BPix1+FPix1_neg+FPix2_neg'
)
PixelLayerTriplets.BPix = cms.PSet(
    TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
    HitProducer = cms.string('siPixelRecHits'),
)    
PixelLayerTriplets.FPix = cms.PSet(
    TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
    HitProducer = cms.string('siPixelRecHits'),
)


