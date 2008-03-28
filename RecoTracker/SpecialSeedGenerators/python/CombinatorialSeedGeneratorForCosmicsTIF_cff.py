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
combinatorialcosmicseedfinderTIF = copy.deepcopy(combinatorialcosmicseedfinder)
combinatorialcosmicseedfinderTIF.UseScintillatorsConstraint = False

