import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HGCalSimProducers.hgcHitAssociation_cfi import lcAssocByEnergyScoreProducer as _lcAssocByEnergyScoreProducer
from SimCalorimetry.HGCalSimProducers.hgcHitAssociation_cfi import scAssocByEnergyScoreProducer as _scAssocByEnergyScoreProducer
from SimCalorimetry.HGCalAssociatorProducers.LCToSCAssociation_cfi import layerClusterSimClusterAssociation as _layerClusterSimClusterAssociationProducer
from SimCalorimetry.HGCalAssociatorProducers.LCToCPAssociation_cfi import layerClusterCaloParticleAssociation as _layerClusterCaloParticleAssociationProducer

from SimCalorimetry.HGCalSimProducers.hgcHitAssociation_cfi import barrelLcAssocByEnergyScoreProducer as _barrelLcAssocByEnergyScoreProducer
from SimCalorimetry.HGCalSimProducers.hgcHitAssociation_cfi import barrelScAssocByEnergyScoreProducer as _barrelScAssocByEnergyScoreProducer
from SimCalorimetry.HGCalAssociatorProducers.LCToSCAssociation_cfi import barrelLayerClusterSimClusterAssociation as _barrelLayerClusterSimClusterAssociation
from SimCalorimetry.HGCalAssociatorProducers.LCToCPAssociation_cfi import barrelLayerClusterCaloParticleAssociation as _barrelLayerClusterCaloParticleAssociation

from SimCalorimetry.HGCalAssociatorProducers.SimClusterToCaloParticleAssociation_cfi import SimClusterToCaloParticleAssociation
from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociation_cfi import  allTrackstersToSimTrackstersAssociationsByLCs as _allTrackstersToSimTrackstersAssociationsByLCs
from SimCalorimetry.HGCalAssociatorProducers.hitToSimClusterCaloParticleAssociator_cfi import hitToSimClusterCaloParticleAssociator as _hitToSimClusterCaloParticleAssociator

from Validation.HGCalValidation.HLT_TICLIterLabels_cff import hltTiclIterLabels as _hltTiclIterLabels

from RecoLocalCalo.HGCalRecProducers.recHitMapProducer_cff import recHitMapProducer as _recHitMapProducer

run3_hits = [
    "hltParticleFlowRecHitECALUnseeded",
    "hltParticleFlowRecHitHBHE"
]

ph2_hits = [
    "hltHGCalRecHit:HGCEERecHits",
    "hltHGCalRecHit:HGCHEFRecHits",
    "hltHGCalRecHit:HGCHEBRecHits",
    "hltParticleFlowRecHitECALUnseeded",
    "hltParticleFlowRecHitHBHE"
]

hltRecHitMapProducer = _recHitMapProducer.clone(
    hits = run3_hits,
    doHgcalHits = False,
    doPFHits = True,
)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(hltRecHitMapProducer, hits=ph2_hits, doHgcalHits=True)

# LC to CP and LC to SC associators TICL-based for HGCal region

hltLcAssocByEnergyScoreProducer = _lcAssocByEnergyScoreProducer.clone(
    hits = cms.InputTag("hltRecHitMapProducer", "RefProdVectorHGCRecHitCollection"),
    hitMapTag = cms.InputTag("hltRecHitMapProducer","hgcalRecHitMap"),
)

hltScAssocByEnergyScoreProducer = _scAssocByEnergyScoreProducer.clone(
    hits = cms.InputTag("hltRecHitMapProducer", "RefProdVectorHGCRecHitCollection"),
    hitMapTag = cms.InputTag("hltRecHitMapProducer","hgcalRecHitMap"),
)

hltLayerClusterCaloParticleAssociationProducer = _layerClusterCaloParticleAssociationProducer.clone(
    associator = cms.InputTag("hltLcAssocByEnergyScoreProducer"),
    label_lc = cms.InputTag("hltMergeLayerClusters")
)

hltLayerClusterSimClusterAssociationProducer = _layerClusterSimClusterAssociationProducer.clone(
    associator = cms.InputTag("hltScAssocByEnergyScoreProducer"),
    label_lcl = cms.InputTag("hltMergeLayerClusters")
)

hltHgcalLayerClustersAssociatorsTask = cms.Task(
    hltLcAssocByEnergyScoreProducer,
    hltScAssocByEnergyScoreProducer,
    hltLayerClusterCaloParticleAssociationProducer,
    hltLayerClusterSimClusterAssociationProducer,
)

# LC to CP and LC to SC associators TICL-based for barrel region (ticl_barrel)

hltBarrelLcAssocByEnergyScoreProducer = _barrelLcAssocByEnergyScoreProducer.clone(
    hits = cms.InputTag("hltRecHitMapProducer", "RefProdVectorPFRecHitCollection"),
    hitMapTag = cms.InputTag("hltRecHitMapProducer","pfRecHitMap"),
)

hltBarrelScAssocByEnergyScoreProducer = _barrelScAssocByEnergyScoreProducer.clone(
    hits = cms.InputTag("hltRecHitMapProducer", "RefProdVectorPFRecHitCollection"),
    hitMapTag = cms.InputTag("hltRecHitMapProducer","pfRecHitMap"),
)

hltBarrelLayerClusterCaloParticleAssociationProducer = _barrelLayerClusterCaloParticleAssociation.clone(
    associator = cms.InputTag("hltBarrelLcAssocByEnergyScoreProducer"),
    label_lc = cms.InputTag("hltBarrelLayerClustersEB"),
    filter_sim_hits = cms.vstring("Ecal",)
)

hltBarrelLayerClusterSimClusterAssociationProducer = _barrelLayerClusterSimClusterAssociation.clone(
    associator = cms.InputTag("hltBarrelScAssocByEnergyScoreProducer"),
    label_lcl = cms.InputTag("hltBarrelLayerClustersEB"),
    filter_sim_hits = cms.vstring("Ecal",)
)

hltHgcalAndBarrelLayerClustersAssociatorsTask = cms.Task(
    hltLcAssocByEnergyScoreProducer,
    hltScAssocByEnergyScoreProducer,
    hltLayerClusterCaloParticleAssociationProducer,
    hltLayerClusterSimClusterAssociationProducer,
    hltBarrelLcAssocByEnergyScoreProducer,
    hltBarrelScAssocByEnergyScoreProducer,
    hltBarrelLayerClusterCaloParticleAssociationProducer,
    hltBarrelLayerClusterSimClusterAssociationProducer,
)

from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
ticl_barrel.toReplaceWith(
    hltHgcalLayerClustersAssociatorsTask,
    hltHgcalAndBarrelLayerClustersAssociatorsTask
)

# LC to Tracksters and Tracksters to SimTracksters associators TICL-based for HGCal region

from SimCalorimetry.HGCalAssociatorProducers.AllLayerClusterToTracksterAssociatorsProducer_cfi import AllLayerClusterToTracksterAssociatorsProducer as _AllLayerClusterToTracksterAssociatorsProducer

hltAllLayerClusterToTracksterAssociations = _AllLayerClusterToTracksterAssociatorsProducer.clone(
    layer_clusters = cms.InputTag("hltMergeLayerClusters"),
    tracksterCollections = cms.VInputTag(
        *[cms.InputTag(label) for label in _hltTiclIterLabels],
        cms.InputTag("hltTiclSimTracksters"),
        cms.InputTag("hltTiclSimTracksters", "fromCPs"),
    )
)

hltAllTrackstersToSimTrackstersAssociationsByLCs = _allTrackstersToSimTrackstersAssociationsByLCs.clone(
    allLCtoTSAccoc =  cms.string("hltAllLayerClusterToTracksterAssociations"),
    layerClusters = cms.InputTag("hltMergeLayerClusters"),
    tracksterCollections = cms.VInputTag(
        *[cms.InputTag(label) for label in _hltTiclIterLabels]
    ),
    simTracksterCollections = cms.VInputTag(
      cms.InputTag('hltTiclSimTracksters'),
      cms.InputTag('hltTiclSimTracksters','fromCPs')
    ),
)

from SimCalorimetry.HGCalAssociatorProducers.AllTracksterToSimTracksterAssociatorsByHitsProducer_cfi import AllTracksterToSimTracksterAssociatorsByHitsProducer as _AllTracksterToSimTracksterAssociatorsByHitsProducer

hltHitToSimClusterCaloParticleAssociator = _hitToSimClusterCaloParticleAssociator.clone(
    hitMap = cms.InputTag("hltRecHitMapProducer","hgcalRecHitMap"),
    hits = cms.InputTag("hltRecHitMapProducer", "RefProdVectorHGCRecHitCollection"),
)

from SimCalorimetry.HGCalAssociatorProducers.AllHitToTracksterAssociatorsProducer_cfi import AllHitToTracksterAssociatorsProducer as _AllHitToTracksterAssociatorsProducer

hltAllHitToTracksterAssociations =  _AllHitToTracksterAssociatorsProducer.clone(
    hitMapTag = cms.InputTag("hltRecHitMapProducer","hgcalRecHitMap"),
    hits = cms.InputTag("hltRecHitMapProducer", "RefProdVectorHGCRecHitCollection"),
    layerClusters = cms.InputTag("hltMergeLayerClusters"),
    tracksterCollections = cms.VInputTag(
        *[cms.InputTag(label) for label in _hltTiclIterLabels],
        cms.InputTag("hltTiclSimTracksters"),
        cms.InputTag("hltTiclSimTracksters", "fromCPs"),
    )
)

hltAllTrackstersToSimTrackstersAssociationsByHits = _AllTracksterToSimTracksterAssociatorsByHitsProducer.clone(
    allHitToTSAccoc = cms.string("hltAllHitToTracksterAssociations"),
    hitToCaloParticleMap = cms.InputTag("hltHitToSimClusterCaloParticleAssociator","hitToCaloParticleMap"),
    hitToSimClusterMap = cms.InputTag("hltHitToSimClusterCaloParticleAssociator","hitToSimClusterMap"),
    hits = cms.InputTag("hltRecHitMapProducer", "RefProdVectorHGCRecHitCollection"),
    tracksterCollections = cms.VInputTag(
        *[cms.InputTag(label) for label in _hltTiclIterLabels]
    ),
    simTracksterCollections = cms.VInputTag(
      'hltTiclSimTracksters',
      'hltTiclSimTracksters:fromCPs'
    ),
)

hltHgcalAssociatorsTask = cms.Task(hltRecHitMapProducer,
                                   SimClusterToCaloParticleAssociation,
                                   hltHgcalLayerClustersAssociatorsTask,
                                   hltAllLayerClusterToTracksterAssociations,
                                   hltAllTrackstersToSimTrackstersAssociationsByLCs,
                                   hltAllHitToTracksterAssociations,
                                   hltHitToSimClusterCaloParticleAssociator,
                                   hltAllTrackstersToSimTrackstersAssociationsByHits
                                   )
