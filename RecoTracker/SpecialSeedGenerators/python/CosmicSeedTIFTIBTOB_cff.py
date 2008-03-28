import FWCore.ParameterSet.Config as cms

# initialize magnetic field #########################
#include "Geometry/CMSCommonData/data/cmsMagneticFieldXML.cfi"
from MagneticField.Engine.uniformMagneticField_cfi import *
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
cosmicseedfinderTIFTIBTOB = copy.deepcopy(cosmicseedfinder)
cosmicseedfinderTIFTIBTOB.GeometricStructure = 'TIBTOB'

