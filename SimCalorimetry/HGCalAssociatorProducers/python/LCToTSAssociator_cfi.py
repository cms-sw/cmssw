import FWCore.ParameterSet.Config as cms

layerClusterToTracksterAssociation = cms.EDProducer("LCToTSAssociatorProducer",
    layer_clusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksters = cms.InputTag("ticlTracksters"),
)

from SimCalorimetry.HGCalAssociatorProducers.LCToTSAssociatorProducer_cfi import LCToTSAssociatorProducer

layerClusterToCLUE3DTracksterAssociation = LCToTSAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersCLUE3DHigh")
)

layerClusterToTracksterMergeAssociation = LCToTSAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersMerge")
)

layerClusterToSimTracksterAssociation = LCToTSAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlSimTracksters")
)

layerClusterToSimTracksterFromCPsAssociation = LCToTSAssociatorProducer.clone(
    tracksters = cms.InputTag("ticlSimTracksters", "fromCPs")
)

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
ticl_v5.toModify(layerClusterToTracksterMergeAssociation, tracksters = cms.InputTag("ticlCandidate"))

from SimCalorimetry.HGCalAssociatorProducers.AllLayerClusterToTracksterAssociatorsProducer_cfi import AllLayerClusterToTracksterAssociatorsProducer
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabels

allLayerClusterToTracksterAssociations = AllLayerClusterToTracksterAssociatorsProducer.clone(    
    tracksterCollections = cms.VInputTag(
        *[cms.InputTag(label) for label in ticlIterLabels],
        cms.InputTag("ticlSimTracksters"),
        cms.InputTag("ticlSimTracksters", "fromCPs"),
    )
)

