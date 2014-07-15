import FWCore.ParameterSet.Config as cms

from RecoTracker.SpecialSeedGenerators.CombinatorialSeedGeneratorForCosmicsRegionalReconstruction_cfi import *

regionalCosmicTrackerSeedingLayers = cms.EDProducer("SeedingLayersEDProducer",
    layerInfo,
    layerList = layerList
)
