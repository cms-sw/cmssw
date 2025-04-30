import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HGCalSimProducers.hgcHitAssociation_cfi import lcAssocByEnergyScoreProducer as _lcAssocByEnergyScoreProducer
from SimCalorimetry.HGCalSimProducers.hgcHitAssociation_cfi import scAssocByEnergyScoreProducer as _scAssocByEnergyScoreProducer
from SimCalorimetry.HGCalAssociatorProducers.LCToSCAssociation_cfi import layerClusterSimClusterAssociation as _layerClusterSimClusterAssociationProducer
from SimCalorimetry.HGCalAssociatorProducers.LCToCPAssociation_cfi import layerClusterCaloParticleAssociation as _layerClusterCaloParticleAssociationProducer

from SimCalorimetry.HGCalAssociatorProducers.SimClusterToCaloParticleAssociation_cfi import SimClusterToCaloParticleAssociation
from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociation_cfi import  allTrackstersToSimTrackstersAssociationsByLCs as _allTrackstersToSimTrackstersAssociationsByLCs
from SimCalorimetry.HGCalAssociatorProducers.hitToSimClusterCaloParticleAssociator_cfi import hitToSimClusterCaloParticleAssociator as _hitToSimClusterCaloParticleAssociator

from Validation.HGCalValidation.HLT_TICLIterLabels_cff import hltTiclIterLabels as _hltTiclIterLabels

from RecoLocalCalo.HGCalRecProducers.recHitMapProducer_cfi import recHitMapProducer as _recHitMapProducer
hltRecHitMapProducer = _recHitMapProducer.clone(
    BHInput = cms.InputTag("hltHGCalRecHit","HGCHEBRecHits"),
    EBInput = cms.InputTag("hltParticleFlowRecHitECALUnseeded"),
    EEInput = cms.InputTag("hltHGCalRecHit","HGCEERecHits"),
    FHInput = cms.InputTag("hltHGCalRecHit","HGCHEFRecHits"),
    HBInput = cms.InputTag("hltParticleFlowRecHitHBHE"),
    HOInput = cms.InputTag("hltParticleFlowRecHitHO"),
    hgcalOnly = cms.bool(True),
)

hltLcAssocByEnergyScoreProducer = _lcAssocByEnergyScoreProducer.clone(
    hits = cms.VInputTag("hltHGCalRecHit:HGCEERecHits", "hltHGCalRecHit:HGCHEFRecHits", "hltHGCalRecHit:HGCHEBRecHits"),
    hitMapTag = cms.InputTag("hltRecHitMapProducer","hgcalRecHitMap"),
)

hltScAssocByEnergyScoreProducer = _scAssocByEnergyScoreProducer.clone(
    hits = cms.VInputTag("hltHGCalRecHit:HGCEERecHits", "hltHGCalRecHit:HGCHEFRecHits", "hltHGCalRecHit:HGCHEBRecHits"),
    hitMapTag = cms.InputTag("hltRecHitMapProducer","hgcalRecHitMap"),
)

hltLayerClusterCaloParticleAssociationProducer = _layerClusterCaloParticleAssociationProducer.clone(
    associator = cms.InputTag("hltLcAssocByEnergyScoreProducer"),
    label_lc = cms.InputTag("hltHgcalMergeLayerClusters")
)

hltLayerClusterSimClusterAssociationProducer = _layerClusterSimClusterAssociationProducer.clone(
    associator = cms.InputTag("hltScAssocByEnergyScoreProducer"),
    label_lcl = cms.InputTag("hltHgcalMergeLayerClusters")
)

from SimCalorimetry.HGCalAssociatorProducers.AllLayerClusterToTracksterAssociatorsProducer_cfi import AllLayerClusterToTracksterAssociatorsProducer as _AllLayerClusterToTracksterAssociatorsProducer

hltAllLayerClusterToTracksterAssociations = _AllLayerClusterToTracksterAssociatorsProducer.clone(
    layer_clusters = cms.InputTag("hltHgcalMergeLayerClusters"),
    tracksterCollections = cms.VInputTag(
        *[cms.InputTag(label) for label in _hltTiclIterLabels],
        cms.InputTag("hltTiclSimTracksters"),
        cms.InputTag("hltTiclSimTracksters", "fromCPs"),
    )
)

hltAllTrackstersToSimTrackstersAssociationsByLCs = _allTrackstersToSimTrackstersAssociationsByLCs.clone(
    allLCtoTSAccoc =  cms.string("hltAllLayerClusterToTracksterAssociations"),
    layerClusters = cms.InputTag("hltHgcalMergeLayerClusters"),
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
    hits = cms.VInputTag("hltHGCalRecHit:HGCEERecHits", "hltHGCalRecHit:HGCHEFRecHits", "hltHGCalRecHit:HGCHEBRecHits")
)

from SimCalorimetry.HGCalAssociatorProducers.AllHitToTracksterAssociatorsProducer_cfi import AllHitToTracksterAssociatorsProducer as _AllHitToTracksterAssociatorsProducer

hltAllHitToTracksterAssociations =  _AllHitToTracksterAssociatorsProducer.clone(
    hitMapTag = cms.InputTag("hltRecHitMapProducer","hgcalRecHitMap"),
    hits = cms.VInputTag("hltHGCalRecHit:HGCEERecHits", "hltHGCalRecHit:HGCHEFRecHits", "hltHGCalRecHit:HGCHEBRecHits"),
    layerClusters = cms.InputTag("hltHgcalMergeLayerClusters"),
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
    hits = cms.VInputTag("hltHGCalRecHit:HGCEERecHits", "hltHGCalRecHit:HGCHEFRecHits", "hltHGCalRecHit:HGCHEBRecHits"),
    tracksterCollections = cms.VInputTag(
        *[cms.InputTag(label) for label in _hltTiclIterLabels]
    ),
    simTracksterCollections = cms.VInputTag(
      'hltTiclSimTracksters',
      'hltTiclSimTracksters:fromCPs'
    ),
)

hltHgcalAssociatorsTask = cms.Task(hltRecHitMapProducer,
                                   hltLcAssocByEnergyScoreProducer,
                                   hltScAssocByEnergyScoreProducer,
                                   SimClusterToCaloParticleAssociation,
                                   hltLayerClusterCaloParticleAssociationProducer,
                                   hltLayerClusterSimClusterAssociationProducer,
                                   hltAllLayerClusterToTracksterAssociations,
                                   hltAllTrackstersToSimTrackstersAssociationsByLCs,
                                   hltAllHitToTracksterAssociations,
                                   hltHitToSimClusterCaloParticleAssociator,
                                   hltAllTrackstersToSimTrackstersAssociationsByHits
                                   )
