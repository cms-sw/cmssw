import FWCore.ParameterSet.Config as cms

# initialize magnetic field #########################
#initialize geometry

#stripCPE
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
#TransientTrackingBuilder
#from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
from RecoTracker.TransientTrackingRecHit.TTRHBuilderWithTemplate_cfi import *
from RecoTracker.SpecialSeedGenerators.CosmicSeed_cfi import *
# generate Cosmic seeds #####################
cosmicseedfinderP5 = cosmicseedfinder.clone(
    GeometricStructure = 'STANDARD',
    HitsForSeeds       = 'pairs'
)
