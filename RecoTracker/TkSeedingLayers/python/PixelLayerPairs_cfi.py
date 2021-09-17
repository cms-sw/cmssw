import FWCore.ParameterSet.Config as cms

from RecoTracker.TkSeedingLayers.seedingLayersEDProducer_cfi import *

PixelLayerPairs = seedingLayersEDProducer.clone(
    layerList = ['BPix1+BPix2', 
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
                 'FPix1_neg+FPix2_neg'],
    BPix = cms.PSet(
	TTRHBuilder = cms.string('WithTrackAngle'),
        HitProducer = cms.string('siPixelRecHits'),
    ),
    FPix = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        HitProducer = cms.string('siPixelRecHits'),
    ),
)
