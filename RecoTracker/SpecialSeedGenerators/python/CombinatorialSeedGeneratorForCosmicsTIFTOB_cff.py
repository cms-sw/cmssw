import FWCore.ParameterSet.Config as cms

# magnetic field

from MagneticField.Engine.uniformMagneticField_cfi import *
# cms geometry
# tracker geometry
# tracker numbering
#stripCPE
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
#pixelCPE
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
#TransientTrackingBuilder
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi import *
import copy
from RecoTracker.SpecialSeedGenerators.CombinatorialSeedGeneratorForCosmics_cfi import *
#recHitMatcher
#include "RecoLocalTracker/SiStripRecHitConverter/data/SiStripRecHitMatcher.cfi"
#seeding module
combinatorialcosmicseedfinderTIFTOB = copy.deepcopy(combinatorialcosmicseedfinder)
combinatorialcosmicseedfinderTIFTOB.OrderedHitsFactoryPSets = cms.VPSet(cms.PSet(
    ComponentName = cms.string('GenericTripletGenerator'),
    LayerPSet = cms.PSet(
        layerInfo,
        layerList = cms.vstring('TOB4+TOB5+TOB6', 
            'TOB3+TOB5+TOB6', 
            'TOB3+TOB4+TOB5', 
            'TOB2+TOB4+TOB5', 
            'TOB2+TOB4+TOB6')
    ),
    PropagationDirection = cms.string('alongMomentum'),
    NavigationDirection = cms.string('outsideIn')
), 
    cms.PSet(
        ComponentName = cms.string('GenericTripletGenerator'),
        LayerPSet = cms.PSet(
            layerInfo,
            layerList = cms.vstring('TOB1+TOB2+TOB3', 
                'TOB1+TOB2+TOB4', 
                'TOB1+TOB2+TOB5')
        ),
        PropagationDirection = cms.string('oppositeToMomentum'),
        NavigationDirection = cms.string('insideOut')
    ))

