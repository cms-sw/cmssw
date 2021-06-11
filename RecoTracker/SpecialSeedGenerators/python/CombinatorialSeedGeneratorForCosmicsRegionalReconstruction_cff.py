import FWCore.ParameterSet.Config as cms

from RecoTracker.SpecialSeedGenerators.CombinatorialSeedGeneratorForCosmicsRegionalReconstruction_cfi import *
import RecoTracker.TkSeedingLayers.seedingLayersEDProducer_cfi as _mod

regionalCosmicTrackerSeedingLayers = _mod.seedingLayersEDProducer.clone(
    layerInfo,
    layerList = layerList
)
