import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.barrelValidator_cfi import barrelValidator as _barrelValidator
from SimCalorimetry.HGCalAssociatorProducers.hltLCToCPAssociation_cfi import hltBarrelLayerClusterCaloParticleAssociation
from SimCalorimetry.HGCalAssociatorProducers.hltLCToSCAssociation_cfi import hltBarrelLayerClusterSimClusterAssociation

hltBarrelValidator = _barrelValidator.clone(
    lclTag = 'hltMergeLayerClusters',
    hits = 'hltRecHitMapProducer:RefProdVectorPFRecHitCollection',
    rechitmapTag = 'hltRecHitMapProducer:barrelRecHitMap',
    associator = ['hltBarrelLayerClusterCaloParticleAssociation',],
    associatorSim = ['hltBarrelLayerClusterSimClusterAssociation',],
    dirName = 'HLT/BarrelCalorimeters/BarrelValidator/'
)

