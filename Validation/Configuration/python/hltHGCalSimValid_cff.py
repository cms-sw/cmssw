import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HGCalAssociatorProducers.SimClusterToCaloParticleAssociation_cfi import SimClusterToCaloParticleAssociation
from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociation_cfi import  allTrackstersToSimTrackstersAssociationsByLCs as _allTrackstersToSimTrackstersAssociationsByLCs

from Validation.HGCalValidation.HLT_TICLIterLabels_cff import hltTiclIterLabelsPSet as _hltTiclIterLabelsPSet

from Validation.Configuration.hltBarrelSimValid_cff import hltBarrelRecHitMapProducer as _hltBarrelRecHitMapProducer
from Validation.Configuration.hltBarrelSimValid_cff import barrel_hits
hgcal_hits = ["hltHGCalRecHit:HGCEERecHits", "hltHGCalRecHit:HGCHEFRecHits", "hltHGCalRecHit:HGCHEBRecHits"]
hltRecHitMapProducer = _hltBarrelRecHitMapProducer.clone()

hltHGCalRecHitMapProducer = _hltBarrelRecHitMapProducer.clone(
    hits = hgcal_hits,
    hgcalOnly = True,
)
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
(phase2_common & ~ticl_barrel).toReplaceWith(hltRecHitMapProducer, hltHGCalRecHitMapProducer)

(phase2_common & ticl_barrel).toModify(hltRecHitMapProducer,
                                       hits = [*hgcal_hits, *barrel_hits],
                                       )

from SimCalorimetry.HGCalAssociatorProducers.AllLayerClusterToTracksterAssociatorsProducer_cfi import AllLayerClusterToTracksterAssociatorsProducer as _AllLayerClusterToTracksterAssociatorsProducer
hltAllLayerClusterToTracksterAssociations = _AllLayerClusterToTracksterAssociatorsProducer.clone(
    layer_clusters = cms.InputTag("hltMergeLayerClusters"),
    tracksterCollections = cms.VInputTag(
        *[cms.InputTag(label) for label in _hltTiclIterLabelsPSet.labels],
        cms.InputTag("hltTiclSimTracksters", "fromBoundarySimCluster"),
        cms.InputTag("hltTiclSimTracksters", "fromCaloParticle"),
    )
)

hltAllTrackstersToSimTrackstersAssociationsByLCs = _allTrackstersToSimTrackstersAssociationsByLCs.clone(
    allLCtoTSAccoc =  cms.string("hltAllLayerClusterToTracksterAssociations"),
    layerClusters = cms.InputTag("hltMergeLayerClusters"),
    tracksterCollections = cms.VInputTag(
        *[cms.InputTag(label) for label in _hltTiclIterLabelsPSet.labels]
    ),
    simTracksterCollections = cms.VInputTag(
        cms.InputTag("hltTiclSimTracksters", "fromBoundarySimCluster"),
        cms.InputTag("hltTiclSimTracksters", "fromCaloParticle"),
    ),
)

from SimCalorimetry.HGCalAssociatorProducers.hitToSimClusterCaloParticleAssociator_cfi import hitToSimClusterCaloParticleAssociator as _hitToSimClusterCaloParticleAssociator
hltHitToSimClusterCaloParticleAssociator = _hitToSimClusterCaloParticleAssociator.clone(
    simClusters = cms.InputTag("mix", "MergedCaloTruthCaloParticle"), # CaloParticle but in SimCluster dataformat
    hitMap = 'hltHGCalRecHitMapProducer:hgcalRecHitMap',
    hits = 'hltHGCalRecHitMapProducer:RefProdVectorHGCRecHitCollection'
)
hltHitToBoundarySimClusterAssociator = _hitToSimClusterCaloParticleAssociator.clone(
    simClusters = cms.InputTag("mix", "MergedCaloTruthBoundaryTrackSimCluster"),
    hitMap = 'hltHGCalRecHitMapProducer:hgcalRecHitMap',
    hits = 'hltHGCalRecHitMapProducer:RefProdVectorHGCRecHitCollection'
)



from SimCalorimetry.HGCalAssociatorProducers.AllHitToTracksterAssociatorsProducer_cfi import AllHitToTracksterAssociatorsProducer as _AllHitToTracksterAssociatorsProducer
hltAllHitToTracksterAssociations =  _AllHitToTracksterAssociatorsProducer.clone(
    hitMapTag = cms.InputTag("hltHGCalRecHitMapProducer","hgcalRecHitMap"),
    hits = cms.InputTag("hltHGCalRecHitMapProducer", "RefProdVectorHGCRecHitCollection"),
    layerClusters = cms.InputTag("hltMergeLayerClusters"),
    tracksterCollections = cms.VInputTag(
        *[cms.InputTag(label) for label in _hltTiclIterLabelsPSet.labels],
        cms.InputTag("hltTiclSimTracksters", "fromBoundarySimCluster"),
        cms.InputTag("hltTiclSimTracksters", "fromCaloParticle"),
    )
)

from SimCalorimetry.HGCalAssociatorProducers.AllTracksterToSimTracksterAssociatorsByHitsProducer_cfi import AllTracksterToSimTracksterAssociatorsByHitsProducer as _AllTracksterToSimTracksterAssociatorsByHitsProducer
hltAllTrackstersToSimTrackstersAssociationsByHits = _AllTracksterToSimTracksterAssociatorsByHitsProducer.clone(
    allHitToTSAccoc = cms.string("hltAllHitToTracksterAssociations"),
    hits = cms.InputTag("hltHGCalRecHitMapProducer", "RefProdVectorHGCRecHitCollection"),
    tracksterCollections = cms.VInputTag(
        *[cms.InputTag(label) for label in _hltTiclIterLabelsPSet.labels]
    ),
    simTracksters = cms.VPSet(
        cms.PSet(
            simTracksterCollection=cms.InputTag("hltTiclSimTracksters", "fromBoundarySimCluster"),
            hitToSimClusterMap=cms.InputTag("hltHitToBoundarySimClusterAssociator")
        ),
        cms.PSet(
            simTracksterCollection=cms.InputTag("hltTiclSimTracksters", "fromCaloParticle"),
            hitToSimClusterMap=cms.InputTag("hltHitToSimClusterCaloParticleAssociator")
        ),
    )
)
from SimCalorimetry.HGCalAssociatorProducers.hltLCToSCAssociation_cfi import hltHGCalLCToSCAssociatorByEnergyScoreProducer, hltHGCalLayerClusterSimClusterAssociation, hltHGCalLayerClusterCaloParticleAssociation

hltHgcalAssociatorsTask = cms.Task(hltHGCalRecHitMapProducer,
                                   hltHGCalLCToSCAssociatorByEnergyScoreProducer,
                                   SimClusterToCaloParticleAssociation,
                                   hltHGCalLayerClusterCaloParticleAssociation,
                                   hltHGCalLayerClusterSimClusterAssociation,
                                   hltAllLayerClusterToTracksterAssociations,
                                   hltAllTrackstersToSimTrackstersAssociationsByLCs,
                                   hltAllHitToTracksterAssociations,
                                   hltHitToBoundarySimClusterAssociator, hltHitToSimClusterCaloParticleAssociator,
                                   hltAllTrackstersToSimTrackstersAssociationsByHits
                                   )

hltHgcalPrevalidation = cms.Sequence(
    hltHGCalLCToSCAssociatorByEnergyScoreProducer *
    hltHGCalLayerClusterCaloParticleAssociation *
    hltHGCalLayerClusterSimClusterAssociation
)
