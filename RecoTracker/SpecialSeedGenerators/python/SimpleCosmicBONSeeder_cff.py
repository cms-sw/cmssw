import FWCore.ParameterSet.Config as cms

from RecoTracker.SpecialSeedGenerators.SimpleCosmicBONSeeder_cfi import *

simpleCosmicBONSeedingLayers = cms.EDProducer("SeedingLayersEDProducer",
    layerInfo,
    layerList = cms.vstring(*layerList)
)
