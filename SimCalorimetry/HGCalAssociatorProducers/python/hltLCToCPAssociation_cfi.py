import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HGCalAssociatorProducers.barrelLCToCPAssociatorByEnergyScoreProducer_cfi import *
hltBarrelLCToCPAssociatorByEnergyScoreProducer = barrelLCToCPAssociatorByEnergyScoreProducer.clone(
    hitMapTag = cms.InputTag('hltRecHitMapProducer', 'barrelRecHitMap'),
    hits = cms.InputTag("hltRecHitMapProducer", "RefProdVectorPFRecHitCollection")
)

from SimCalorimetry.HGCalSimProducers.hgcHitAssociation_cfi import lcAssocByEnergyScoreProducer
hltHGCalLCToCPAssociatorByEnergyScoreProducer = lcAssocByEnergyScoreProducer.clone(
    hitMapTag = cms.InputTag('hltRecHitMapProducer', 'hgcalRecHitMap'),
    hits = cms.InputTag("hltRecHitMapProducer", "RefProdVectorHGCRecHitCollection")
)

from SimCalorimetry.HGCalAssociatorProducers.LCToCPAssociation_cfi import barrelLayerClusterCaloParticleAssociation as _barrelLayerClusterCaloParticleAssociation
hltBarrelLayerClusterCaloParticleAssociation = _barrelLayerClusterCaloParticleAssociation.clone(
    associator = cms.InputTag('hltBarrelLCToCPAssociatorByEnergyScoreProducer'),
    label_lc = cms.InputTag('hltMergeLayerClusters')
)
from SimCalorimetry.HGCalAssociatorProducers.LCToCPAssociation_cfi import layerClusterCaloParticleAssociation as _layerClusterCaloParticleAssociation
hltHGCalLayerClusterCaloParticleAssociation = _layerClusterCaloParticleAssociation.clone(
    associator = cms.InputTag('hltHGCalLCToCPAssociatorByEnergyScoreProducer'),
    label_lc = cms.InputTag('hltMergeLayerClusters')
)
