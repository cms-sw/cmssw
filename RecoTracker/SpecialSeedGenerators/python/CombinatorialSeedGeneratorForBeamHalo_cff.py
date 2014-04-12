import FWCore.ParameterSet.Config as cms

# get TTRHBuilderWithoutAngle4PixelPairs
from RecoTracker.TkSeedingLayers.PixelLayerPairs_cfi import *
#get the module combinatorialbeamhaloseedfinder
from RecoTracker.SpecialSeedGenerators.CombinatorialSeedGeneratorForBeamHalo_cfi import *

beamhaloTrackerSeedingLayers = cms.EDProducer("SeedingLayersEDProducer",
    layerInfo,
    layerList = layerList
)
