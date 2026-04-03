import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HGCalAssociatorProducers.hltLCToCPAssociation_cfi import (hltBarrelLCToCPAssociatorByEnergyScoreProducer,
                                                                              hltBarrelLayerClusterCaloParticleAssociation)
from SimCalorimetry.HGCalAssociatorProducers.hltLCToSCAssociation_cfi import (hltBarrelLCToSCAssociatorByEnergyScoreProducer,
                                                                              hltBarrelLayerClusterSimClusterAssociation)

hltBarrelPrevalidation = cms.Sequence(
    hltBarrelLCToCPAssociatorByEnergyScoreProducer *
    hltBarrelLCToSCAssociatorByEnergyScoreProducer *
    hltBarrelLayerClusterCaloParticleAssociation *
    hltBarrelLayerClusterSimClusterAssociation
)
