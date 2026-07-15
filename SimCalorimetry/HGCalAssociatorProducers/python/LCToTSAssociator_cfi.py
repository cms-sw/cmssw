import FWCore.ParameterSet.Config as cms

# the "allLayerClusterToTracksterAssociations" is now used, the individual-producer version (LCToTSAssociatorProducer) is not used anymore
layerClusterToTracksterAssociation = cms.EDProducer("LCToTSAssociatorProducer",
    layer_clusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksters = cms.InputTag("ticlTracksters"),
)

from SimCalorimetry.HGCalAssociatorProducers.LCToTSAssociatorProducer_cfi import LCToTSAssociatorProducer

layerClusterToCLUE3DTracksterAssociation = LCToTSAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersCLUE3DHigh")
)

layerClusterToTracksterMergeAssociation = LCToTSAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlCandidate")
)

layerClusterToSimTracksterAssociation = LCToTSAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlSimTracksters", "fromLegacySimCluster")
)

layerClusterToSimTracksterFromCPsAssociation = LCToTSAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlSimTracksters", "fromCaloParticle")
)

## Barrel
barrelLayerClusterToTracksterAssociation = LCToTSAssociatorProducer.clone(
    tracksters = cms.InputTag('ticlBarrelTracksters')
)

barrelLayerClusterToSimTracksterAssociation = LCToTSAssociatorProducer.clone(
    tracksters = cms.InputTag('ticlBarrelSimTracksters', "fromLegacySimCluster")
)

barrelLayerClusterToSimTracksterFromCPsAssociation = LCToTSAssociatorProducer.clone(
    tracksters = cms.InputTag('ticlBarrelSimTracksters', 'fromCaloParticle')
)

from SimCalorimetry.HGCalAssociatorProducers.AllLayerClusterToTracksterAssociatorsProducer_cfi import AllLayerClusterToTracksterAssociatorsProducer
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabelsPSet

allLayerClusterToTracksterAssociations = AllLayerClusterToTracksterAssociatorsProducer.clone(    
    tracksterCollections = cms.VInputTag(
        *[cms.InputTag(label) for label in ticlIterLabelsPSet.labels],
        cms.InputTag("ticlSimTracksters", "fromBoundarySimCluster"),
        cms.InputTag("ticlSimTracksters", "fromCaloParticle"),
    )
)

allBarrelLayerClusterToTracksterAssociations = AllLayerClusterToTracksterAssociatorsProducer.clone(
    layer_clusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksterCollections = cms.VInputTag(cms.InputTag("ticlTrackstersCLUE3DBarrel"), cms.InputTag("ticlSimTrackstersBarrel", "fromBoundarySimCluster"), cms.InputTag("ticlSimTrackstersBarrel", "fromCaloParticle"))
)
