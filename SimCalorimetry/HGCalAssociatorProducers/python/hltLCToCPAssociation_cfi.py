import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HGCalAssociatorProducers.barrelLCToCPAssociatorByEnergyScoreProducer_cfi import barrelLCToCPAssociatorByEnergyScoreProducer as _barrelLCToCPAssociatorByEnergyScoreProducer
hltBarrelLCToCPAssociatorByEnergyScoreProducer = _barrelLCToCPAssociatorByEnergyScoreProducer.clone(
    hitMapTag = 'hltRecHitMapProducer:barrelRecHitMap',
    hits = 'hltRecHitMapProducer:RefProdVectorPFRecHitCollection'
)

from SimCalorimetry.HGCalSimProducers.hgcHitAssociation_cfi import lcAssocByEnergyScoreProducer as _lcAssocByEnergyScoreProducer
hltHGCalLCToCPAssociatorByEnergyScoreProducer = _lcAssocByEnergyScoreProducer.clone(
    hitMapTag = 'hltRecHitMapProducer:hgcalRecHitMap',
    hits = 'hltRecHitMapProducer:RefProdVectorHGCRecHitCollection'
)

from SimCalorimetry.HGCalAssociatorProducers.LCToCPAssociation_cfi import barrelLayerClusterCaloParticleAssociation as _barrelLayerClusterCaloParticleAssociation
hltBarrelLayerClusterCaloParticleAssociation = _barrelLayerClusterCaloParticleAssociation.clone(
    associator = 'hltBarrelLCToCPAssociatorByEnergyScoreProducer',
    label_lc = 'hltMergeLayerClusters'
)

from SimCalorimetry.HGCalAssociatorProducers.LCToCPAssociation_cfi import layerClusterCaloParticleAssociation as _layerClusterCaloParticleAssociation
hltHGCalLayerClusterCaloParticleAssociation = _layerClusterCaloParticleAssociation.clone(
    associator = 'hltHGCalLCToCPAssociatorByEnergyScoreProducer',
    label_lc = 'hltMergeLayerClusters'
)
