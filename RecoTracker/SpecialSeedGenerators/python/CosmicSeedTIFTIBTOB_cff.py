import FWCore.ParameterSet.Config as cms

# initialize magnetic field #########################

from MagneticField.Engine.uniformMagneticField_cfi import *
#initialize geometry

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

