import FWCore.ParameterSet.Config as cms

from RecoTracker.TkSeedingLayers.seedingLayersEDProducer_cfi import *

PixelLayerTriplets = seedingLayersEDProducer.clone(
    layerList = ['BPix1+BPix2+BPix3', 
                 'BPix1+BPix2+FPix1_pos', 
                 'BPix1+BPix2+FPix1_neg', 
                 'BPix1+FPix1_pos+FPix2_pos', 
                 'BPix1+FPix1_neg+FPix2_neg'],
    BPix = cms.PSet(
	TTRHBuilder = cms.string('WithTrackAngle'),
        HitProducer = cms.string('siPixelRecHits'),
    ),    
    FPix = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        HitProducer = cms.string('siPixelRecHits'),
    )
)
_layersForPhase1 = [
    'BPix1+BPix2+BPix3',
    'BPix2+BPix3+BPix4',
    'BPix1+BPix3+BPix4',
    'BPix1+BPix2+BPix4',
    'BPix2+BPix3+FPix1_pos',
    'BPix2+BPix3+FPix1_neg',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix2+FPix1_pos+FPix2_pos',
    'BPix2+FPix1_neg+FPix2_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg',
    'FPix1_pos+FPix2_pos+FPix3_pos',
    'FPix1_neg+FPix2_neg+FPix3_neg'
]
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(PixelLayerTriplets, layerList=_layersForPhase1)

_layersForPhase2 = [ 'BPix1+BPix2+BPix3', 'BPix2+BPix3+BPix4',
                     'BPix2+BPix3+FPix1_pos', 'BPix2+BPix3+FPix1_neg',
                     'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg',
                     'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg',
                     'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg',
                     'BPix1+FPix2_pos+FPix3_pos', 'BPix1+FPix2_neg+FPix3_neg',
                     'FPix1_pos+FPix2_pos+FPix3_pos', 'FPix1_neg+FPix2_neg+FPix3_neg',
                     'FPix2_pos+FPix3_pos+FPix4_pos', 'FPix2_neg+FPix3_neg+FPix4_neg',
                     'FPix3_pos+FPix4_pos+FPix5_pos', 'FPix3_neg+FPix4_neg+FPix5_neg',
                     'FPix4_pos+FPix5_pos+FPix6_pos', 'FPix4_neg+FPix5_neg+FPix6_neg',
                     'FPix5_pos+FPix6_pos+FPix7_pos', 'FPix5_neg+FPix6_neg+FPix7_neg',
                     'FPix6_pos+FPix7_pos+FPix8_pos', 'FPix6_neg+FPix7_neg+FPix8_neg',
                     'FPix6_pos+FPix7_pos+FPix9_pos', 'FPix6_neg+FPix7_neg+FPix9_neg'
]
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(PixelLayerTriplets, layerList=_layersForPhase2)
