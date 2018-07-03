import FWCore.ParameterSet.Config as cms

from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import PixelLayerTriplets as _PixelLayerTriplets

PixelLayerQuadruplets = _PixelLayerTriplets.clone(
    layerList = [
        ## straightforward combinations:
        'BPix1+BPix2+BPix3+BPix4',
        'BPix1+BPix2+BPix3+FPix1_pos',
        'BPix1+BPix2+BPix3+FPix1_neg',
        'BPix1+BPix2+FPix1_pos+FPix2_pos',
        'BPix1+BPix2+FPix1_neg+FPix2_neg',
        'BPix1+FPix1_pos+FPix2_pos+FPix3_pos',
        'BPix1+FPix1_neg+FPix2_neg+FPix3_neg'
#        ## "gap" combinations:
#        'BPix2+FPix1_pos+FPix2_pos+FPix3_pos',
#        'BPix2+FPix1_neg+FPix2_neg+FPix3_neg',
#        'BPix1+BPix2+FPix2_pos+FPix3_pos',
#        'BPix1+BPix2+FPix2_neg+FPix3_neg',
#        'BPix1+BPix2+FPix1_pos+FPix3_pos',
#        'BPix1+BPix2+FPix1_neg+FPix3_neg'
    ]
)

_layerListForPhase2 = ['BPix1+BPix2+BPix3+BPix4',
                       'BPix1+BPix2+BPix3+FPix1_pos','BPix1+BPix2+BPix3+FPix1_neg',
                       'BPix1+BPix2+FPix1_pos+FPix2_pos', 'BPix1+BPix2+FPix1_neg+FPix2_neg',
                       'BPix1+FPix1_pos+FPix2_pos+FPix3_pos', 'BPix1+FPix1_neg+FPix2_neg+FPix3_neg',
                       'FPix1_pos+FPix2_pos+FPix3_pos+FPix4_pos', 'FPix1_neg+FPix2_neg+FPix3_neg+FPix4_neg',
                       'FPix2_pos+FPix3_pos+FPix4_pos+FPix5_pos', 'FPix2_neg+FPix3_neg+FPix4_neg+FPix5_neg',
                       'FPix3_pos+FPix4_pos+FPix5_pos+FPix6_pos', 'FPix3_neg+FPix4_neg+FPix5_neg+FPix6_neg',
                       'FPix4_pos+FPix5_pos+FPix6_pos+FPix7_pos', 'FPix4_neg+FPix5_neg+FPix6_neg+FPix7_neg',
                       'FPix5_pos+FPix6_pos+FPix7_pos+FPix8_pos', 'FPix5_neg+FPix6_neg+FPix7_neg+FPix8_neg',
#  removed as redunant and covering effectively only eta>4   (here for documentation, to be optimized after TDR)
#                       'FPix5_pos+FPix6_pos+FPix7_pos+FPix9_pos', 'FPix5_neg+FPix6_neg+FPix7_neg+FPix9_neg',
#                       'FPix6_pos+FPix7_pos+FPix8_pos+FPix9_pos', 'FPix6_neg+FPix7_neg+FPix8_neg+FPix9_neg',
#                       'FPix8_pos+FPix9_pos+FPix10_pos+FPix11_pos', 'FPix8_neg+FPix9_neg+FPix10_neg+FPix11_neg',
#                        'FPix11_pos'FPix9_pos+FPix10_pos+FPix12_pos', 'FPix9_neg+FPix10_neg+FPix11_neg+FPix12_neg'
]

from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(PixelLayerQuadruplets, layerList = _layerListForPhase2)
