import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HGCalAssociatorProducers.barrelLCToSCAssociatorByEnergyScoreProducer_cfi import *
hltBarrelLCToSCAssociatorByEnergyScoreProducer = barrelLCToSCAssociatorByEnergyScoreProducer.clone(
    hitMapTag = 'hltRecHitMapProducer:barrelRecHitMap',
    hits = 'hltRecHitMapProducer:RefProdVectorPFRecHitCollection'
)

from SimCalorimetry.HGCalSimProducers.hgcHitAssociation_cfi import scAssocByEnergyScoreProducer
hltHGCalLCToSCAssociatorByEnergyScoreProducer = scAssocByEnergyScoreProducer.clone(
    hitMapTag = 'hltRecHitMapProducer:hgcalRecHitMap',
    hits = 'hltRecHitMapProducer:RefProdVectorHGCRecHitCollection'
)

from SimCalorimetry.HGCalAssociatorProducers.LCToSCAssociation_cfi import barrelLayerClusterSimClusterAssociation as _barrelLayerClusterSimClusterAssociation
hltBarrelLayerClusterSimClusterAssociation = _barrelLayerClusterSimClusterAssociation.clone(
    associator = 'hltBarrelLCToSCAssociatorByEnergyScoreProducer',
    label_lcl = 'hltMergeLayerClusters'
)
from SimCalorimetry.HGCalAssociatorProducers.LCToSCAssociation_cfi import layerClusterSimClusterAssociation as _layerClusterSimClusterAssociation
hltHGCalLayerClusterSimClusterAssociation = _layerClusterSimClusterAssociation.clone(
    associator = 'hltHGCalLCToSCAssociatorByEnergyScoreProducer',
    label_lcl = 'hltMergeLayerClusters'
)
