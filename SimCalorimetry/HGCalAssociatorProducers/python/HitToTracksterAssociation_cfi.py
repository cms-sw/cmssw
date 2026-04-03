import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HGCalAssociatorProducers.hitToTracksterAssociator_cfi import hitToTracksterAssociator

hitToTrackstersAssociationLinking = hitToTracksterAssociator.clone(
    tracksters = cms.InputTag("ticlTrackstersMerge"),
)


hitToTrackstersAssociationPR = hitToTracksterAssociator.clone(
    tracksters = cms.InputTag("ticlTrackstersCLUE3DHigh"),
)

hitToSimTracksterAssociation = hitToTracksterAssociator.clone(
    tracksters = cms.InputTag("ticlSimTracksters"),
)

hitToSimTracksterFromCPsAssociation = hitToTracksterAssociator.clone(
    tracksters = cms.InputTag("ticlSimTracksters", "fromCPs"),
)


from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5

ticl_v5.toModify(hitToTrackstersAssociationLinking, tracksters = cms.InputTag("ticlCandidate"))

from SimCalorimetry.HGCalAssociatorProducers.AllHitToTracksterAssociatorsProducer_cfi import AllHitToTracksterAssociatorsProducer
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabels

allHitToTracksterAssociations = AllHitToTracksterAssociatorsProducer.clone(    
    tracksterCollections = cms.VInputTag(
        *[cms.InputTag(label) for label in ticlIterLabels],
        cms.InputTag("ticlSimTracksters"),
        cms.InputTag("ticlSimTracksters", "fromCPs"),
    )
)

## Barrel
from SimCalorimetry.HGCalAssociatorProducers.AllHitToBarrelTracksterAssociatorsProducer_cfi import AllHitToBarrelTracksterAssociatorsProducer

allHitToBarrelTracksterAssociations = AllHitToBarrelTracksterAssociatorsProducer.clone(
    layerClusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksterCollections = cms.VInputTag('ticlTrackstersCLUE3DBarrel', 'ticlSimTrackstersBarrel', 'ticlSimTrackstersBarrel:fromCPs')
)
