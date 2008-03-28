import FWCore.ParameterSet.Config as cms

# initialize magnetic field #########################
from Geometry.CMSCommonData.cmsMagneticFieldXML_cfi import *
#initialize geometry
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
#stripCPE
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
#pixelCPE
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
#TransientTrackingBuilder
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
import copy
from RecoTracker.SpecialSeedGenerators.CosmicSeed_cfi import *
# generate Cosmic seeds #####################
cosmicseedfinderP5 = copy.deepcopy(cosmicseedfinder)
cosmicseedfinderP5.GeometricStructure = 'TECPAIRS_TOBTRIPLETS'
cosmicseedfinderP5.HitsForSeeds = 'pairsandtriplets'

