import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecProducers.recHitMapProducer_cff import recHitMapProducer as _recHitMapProducer
barrel_hits = ["hltParticleFlowRecHitECALUnseeded", "hltParticleFlowRecHitHBHE"]
hltBarrelRecHitMapProducer = _recHitMapProducer.clone(
    hits = barrel_hits,
    hgcalOnly = False,
)

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
