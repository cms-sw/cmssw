import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HGCalAssociatorProducers.barrelLCToSCAssociatorByEnergyScoreProducer_cfi import *
hltBarrelLCToSCAssociatorByEnergyScoreProducer = barrelLCToSCAssociatorByEnergyScoreProducer.clone(
    hitMapTag = cms.InputTag('hltRecHitMapProducer', 'barrelRecHitMap'),
    hits = cms.InputTag("hltRecHitMapProducer", "RefProdVectorPFRecHitCollection")
)

from SimCalorimetry.HGCalSimProducers.hgcHitAssociation_cfi import scAssocByEnergyScoreProducer
hltHGCalLCToSCAssociatorByEnergyScoreProducer = scAssocByEnergyScoreProducer.clone(
    hitMapTag = cms.InputTag('hltRecHitMapProducer', 'hgcalRecHitMap'),
    hits = cms.InputTag("hltRecHitMapProducer", "RefProdVectorHGCRecHitCollection")
)

from SimCalorimetry.HGCalAssociatorProducers.LCToSCAssociation_cfi import barrelLayerClusterSimClusterAssociation as _barrelLayerClusterSimClusterAssociation
hltBarrelLayerClusterSimClusterAssociation = _barrelLayerClusterSimClusterAssociation.clone(
    associator = cms.InputTag('hltBarrelLCToSCAssociatorByEnergyScoreProducer'),
    label_lcl = cms.InputTag('hltMergeLayerClusters')
)
from SimCalorimetry.HGCalAssociatorProducers.LCToSCAssociation_cfi import layerClusterSimClusterAssociation as _layerClusterSimClusterAssociation
hltHGCalLayerClusterSimClusterAssociation = _layerClusterSimClusterAssociation.clone(
    associator = cms.InputTag('hltHGCalLCToSCAssociatorByEnergyScoreProducer'),
    label_lcl = cms.InputTag('hltMergeLayerClusters')
)
