import FWCore.ParameterSet.Config as cms

# initialize magnetic field #########################
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
cosmicseedfinderP5 = copy.deepcopy(cosmicseedfinder)
cosmicseedfinderP5.GeometricStructure = 'TECPAIRS_TOBTRIPLETS'
cosmicseedfinderP5.HitsForSeeds = 'pairsandtriplets'

