import FWCore.ParameterSet.Config as cms

from RecoTracker.TkSeedingLayers.seedingLayersEDProducer_cfi import *

PixelLayerPairs = seedingLayersEDProducer.clone()
PixelLayerPairs.layerList = cms.vstring('BPix1+BPix2', 
        'BPix1+BPix3', 
        'BPix2+BPix3', 
        'BPix1+FPix1_pos', 
        'BPix1+FPix1_neg', 
        'BPix1+FPix2_pos', 
        'BPix1+FPix2_neg', 
        'BPix2+FPix1_pos', 
        'BPix2+FPix1_neg', 
        'BPix2+FPix2_pos', 
        'BPix2+FPix2_neg', 
        'FPix1_pos+FPix2_pos', 
        'FPix1_neg+FPix2_neg'
)
PixelLayerPairs.BPix = cms.PSet(
    TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
    HitProducer = cms.string('siPixelRecHits'),
)
PixelLayerPairs.FPix = cms.PSet(
    TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
    HitProducer = cms.string('siPixelRecHits'),
)



