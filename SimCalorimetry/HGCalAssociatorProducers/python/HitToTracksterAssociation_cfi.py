import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HGCalAssociatorProducers.hitToTracksterAssociator_cfi import hitToTracksterAssociator as _hitToTracksterAssociator

# the "single" hitToTrackstersAssociation are not used (only the allHitToTracksterAssociations one is used)
hitToTrackstersAssociationLinking = _hitToTracksterAssociator.clone(
    tracksters = cms.InputTag("ticlCandidate"),
)


hitToTrackstersAssociationPR = _hitToTracksterAssociator.clone(
    tracksters = cms.InputTag("ticlTrackstersCLUE3DHigh"),
)

hitToSimTracksterAssociation = _hitToTracksterAssociator.clone(
    tracksters = cms.InputTag("ticlSimTracksters", "fromLegacySimCluster"),
)

hitToSimTracksterFromCPsAssociation = _hitToTracksterAssociator.clone(
    tracksters = cms.InputTag("ticlSimTracksters", "fromCaloParticle"),
)


from SimCalorimetry.HGCalAssociatorProducers.AllHitToTracksterAssociatorsProducer_cfi import AllHitToTracksterAssociatorsProducer as _AllHitToTracksterAssociatorsProducer
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabelsPSet

allHitToTracksterAssociations = _AllHitToTracksterAssociatorsProducer.clone(    
    tracksterCollections = cms.VInputTag(
        *[cms.InputTag(label) for label in ticlIterLabelsPSet.labels],
        cms.InputTag("ticlSimTracksters", "fromBoundarySimCluster"),
        cms.InputTag("ticlSimTracksters", "fromCaloParticle"),
    )
)

## Barrel
from SimCalorimetry.HGCalAssociatorProducers.AllHitToBarrelTracksterAssociatorsProducer_cfi import AllHitToBarrelTracksterAssociatorsProducer

allHitToBarrelTracksterAssociations = AllHitToBarrelTracksterAssociatorsProducer.clone(
    layerClusters = cms.InputTag("hgcalMergeLayerClusters"),
    tracksterCollections = cms.VInputTag('ticlTrackstersCLUE3DBarrel', 'ticlSimTrackstersBarrel:fromBoundarySimCluster', 'ticlSimTrackstersBarrel:fromCaloParticle')
)
