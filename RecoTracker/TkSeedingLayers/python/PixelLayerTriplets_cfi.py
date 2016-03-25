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
    TTRHBuilder = cms.string('WithTrackAngle'),
    HitProducer = cms.string('siPixelRecHits'),
)    
PixelLayerTriplets.FPix = cms.PSet(
    TTRHBuilder = cms.string('WithTrackAngle'),
    HitProducer = cms.string('siPixelRecHits'),
)

from Configuration.StandardSequences.Eras import eras
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
eras.trackingPhase1.toModify(PixelLayerTriplets, layerList=_layersForPhase1)
eras.trackingPhase1PU70.toModify(PixelLayerTriplets, layerList=_layersForPhase1)
