import FWCore.ParameterSet.Config as cms

# get TTRHBuilderWithoutAngle4PixelPairs
from RecoTracker.TkSeedingLayers.PixelLayerPairs_cfi import *
#get the module combinatorialbeamhaloseedfinder
from RecoTracker.SpecialSeedGenerators.CombinatorialSeedGeneratorForBeamHalo_cfi import *
import RecoTracker.TkSeedingLayers.seedingLayersEDProducer_cfi as _mod

beamhaloTrackerSeedingLayers = _mod.seedingLayersEDProducer.clone(
    layerInfo,
    layerList = layerList,
)
