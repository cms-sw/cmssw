import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HGCalAssociatorProducers.HitToTracksterAssociation_cfi import *
from SimCalorimetry.HGCalAssociatorProducers.tracksterToSimTracksterAssociatorByHitsProducer_cfi import tracksterToSimTracksterAssociatorByHitsProducer



tracksterSimTracksterAssociationByHitsLinking = tracksterToSimTracksterAssociatorByHitsProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersMerge"),
    hitToTracksterMap = cms.InputTag("allHitToTracksterAssociations","hitTo"+"ticlTrackstersMerge"),
    tracksterToHitMap = cms.InputTag("allHitToTracksterAssociations","ticlTrackstersMerge"+"ToHit"),
    hitToSimTracksterMap = cms.InputTag("allHitToTracksterAssociations", "hitTo"+"ticlSimTracksters"),
    hitToSimTracksterFromCPMap = cms.InputTag("allHitToTracksterAssociations", 'hitTo'+'ticlSimTracksters'+'fromCPs'),
    simTracksterToHitMap = cms.InputTag('allHitToTracksterAssociations', 'ticlSimTracksters'+'ToHit'),
    simTracksterFromCPToHitMap = cms.InputTag('allHitToTracksterAssociations', 'ticlSimTracksters'+'fromCPs'+'ToHit'),
)


tracksterSimTracksterAssociationByHitsPR = tracksterToSimTracksterAssociatorByHitsProducer.clone(
    tracksters = cms.InputTag("ticlTrackstersCLUE3DHigh"),
    hitToTracksterMap = cms.InputTag("allHitToTracksterAssociations","hitTo"+"ticlTrackstersCLUE3DHigh"),
    tracksterToHitMap = cms.InputTag("allHitToTracksterAssociations","ticlTrackstersCLUE3DHigh"+"ToHit"),
)



from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5

ticl_v5.toModify(tracksterSimTracksterAssociationByHitsLinking, tracksters = cms.InputTag("ticlCandidate"), hitToTracksterMap = cms.InputTag("allHitToTracksterAssociations","hitTo"+"ticlCandidate"), tracksterToHitMap = cms.InputTag("allHitToTracksterAssociations","ticlCandidate"+"ToHit"))



from SimCalorimetry.HGCalAssociatorProducers.AllTracksterToSimTracksterAssociatorsByHitsProducer_cfi import AllTracksterToSimTracksterAssociatorsByHitsProducer
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabels

allTrackstersToSimTrackstersAssociationsByHits = AllTracksterToSimTracksterAssociatorsByHitsProducer.clone(    
    tracksterCollections = cms.VInputTag(
        *[cms.InputTag(label) for label in ticlIterLabels]
    ),
    simTracksterCollections = cms.VInputTag(
      'ticlSimTracksters',
      'ticlSimTracksters:fromCPs'
    ),
)


from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2

premix_stage2.toModify(tracksterSimTracksterAssociationByHitsLinking,
    caloParticles = "mixData:MergedCaloTruth",
)

premix_stage2.toModify(tracksterSimTracksterAssociationByHitsPR,
    caloParticles = "mixData:MergedCaloTruth",
)

premix_stage2.toModify(allTrackstersToSimTrackstersAssociationsByHits,
    caloParticles = "mixData:MergedCaloTruth",
)
