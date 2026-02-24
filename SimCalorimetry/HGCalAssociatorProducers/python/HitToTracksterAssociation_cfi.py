import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HGCalAssociatorProducers.hitToTracksterAssociator_cfi import hitToTracksterAssociator

hitToTrackstersAssociationLinking = hitToTracksterAssociator.clone(
    tracksters = cms.InputTag("ticlCandidate"),
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


from SimCalorimetry.HGCalAssociatorProducers.AllHitToTracksterAssociatorsProducer_cfi import AllHitToTracksterAssociatorsProducer
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabels

allHitToTracksterAssociations = AllHitToTracksterAssociatorsProducer.clone(    
    tracksterCollections = cms.VInputTag(
        *[cms.InputTag(label) for label in ticlIterLabels],
        cms.InputTag("ticlSimTracksters"),
        cms.InputTag("ticlSimTracksters", "fromCPs"),
    )
)


