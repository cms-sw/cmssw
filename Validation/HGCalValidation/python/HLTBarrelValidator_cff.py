import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.barrelValidator_cfi import barrelValidator as _barrelValidator
from SimCalorimetry.HGCalAssociatorProducers.hltLCToSCAssociation_cfi import hltBarrelLayerClusterSimClusterAssociation, hltBarrelLayerClusterSimClusterAssociation

hltBarrelValidator = _barrelValidator.clone(
    lclTag = 'hltMergeLayerClusters',
    hits = 'hltRecHitMapProducer:RefProdVectorPFRecHitCollection',
    rechitmapTag = 'hltRecHitMapProducer:barrelRecHitMap',
    associator = ['hltBarrelLayerClusterCaloParticleAssociation',],
    associatorSim = ['hltBarrelLayerClusterSimClusterAssociation',],
    doTrackstersPlots = False,
    dirName = 'HLT/BarrelCalorimeters/BarrelValidator/'
)

