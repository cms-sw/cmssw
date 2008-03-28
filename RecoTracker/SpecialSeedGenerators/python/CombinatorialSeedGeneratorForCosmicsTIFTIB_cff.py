import FWCore.ParameterSet.Config as cms

# magnetic field
#include "Geometry/CMSCommonData/data/cmsMagneticFieldXML.cfi"
from MagneticField.Engine.uniformMagneticField_cfi import *
# cms geometry
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
# tracker geometry
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
# tracker numbering
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
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
combinatorialcosmicseedfinderTIFTIB = copy.deepcopy(combinatorialcosmicseedfinder)
combinatorialcosmicseedfinderTIFTIB.OrderedHitsFactoryPSets = cms.VPSet(cms.PSet(
    ComponentName = cms.string('GenericPairGenerator'),
    LayerPSet = cms.PSet(
        layerInfo,
        layerList = cms.vstring('TIB3+TIB4')
    ),
    PropagationDirection = cms.string('alongMomentum'),
    NavigationDirection = cms.string('outsideIn')
), cms.PSet(
    ComponentName = cms.string('GenericPairGenerator'),
    LayerPSet = cms.PSet(
        layerInfo,
        layerList = cms.vstring('TIB1+TIB2')
    ),
    PropagationDirection = cms.string('oppositeToMomentum'),
    NavigationDirection = cms.string('insideOut')
))
combinatorialcosmicseedfinderTIFTIB.SeedMomentum = 0.2
combinatorialcosmicseedfinderTIFTIB.UpperScintillatorParameters.WidthInX = 50
combinatorialcosmicseedfinderTIFTIB.UpperScintillatorParameters.LenghtInZ = 200
combinatorialcosmicseedfinderTIFTIB.UpperScintillatorParameters.GlobalX = 20
combinatorialcosmicseedfinderTIFTIB.UpperScintillatorParameters.GlobalY = 170
combinatorialcosmicseedfinderTIFTIB.UpperScintillatorParameters.GlobalZ = 50
combinatorialcosmicseedfinderTIFTIB.LowerScintillatorParameters.WidthInX = 50
combinatorialcosmicseedfinderTIFTIB.LowerScintillatorParameters.LenghtInZ = 200
combinatorialcosmicseedfinderTIFTIB.LowerScintillatorParameters.GlobalX = 0
combinatorialcosmicseedfinderTIFTIB.LowerScintillatorParameters.GlobalY = -100
combinatorialcosmicseedfinderTIFTIB.LowerScintillatorParameters.GlobalZ = 50

