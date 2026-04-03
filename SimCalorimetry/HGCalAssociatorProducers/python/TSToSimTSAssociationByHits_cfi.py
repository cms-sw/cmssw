import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HGCalAssociatorProducers.HitToTracksterAssociation_cfi import *
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

premix_stage2.toModify(allTrackstersToSimTrackstersAssociationsByHits,
    caloParticles = "mixData:MergedCaloTruth",
)

## Barrel
from SimCalorimetry.HGCalAssociatorProducers.AllBarrelTracksterToSimTracksterAssociatorsByHitsProducer_cfi import AllBarrelTracksterToSimTracksterAssociatorsByHitsProducer

allBarrelTrackstersToSimTrackstersAssociationsByHits = AllBarrelTracksterToSimTracksterAssociatorsByHitsProducer.clone(
    allHitToTSAccoc = cms.string('allHitToBarrelTracksterAssociations'),
    hitToCaloParticleMap = cms.InputTag("barrelHitToSimClusterCaloParticleAssociator", "hitToCaloParticleMap"),
    hitToSimClusterMap = cms.InputTag("barrelHitToSimClusterCaloParticleAssociator", "hitToSimClusterMap"),
    tracksterCollections = cms.VInputTag('ticlTrackstersCLUE3DBarrel'),
    simTracksterCollections = cms.VInputTag(
        'ticlSimTrackstersBarrel',
        'ticlSimTrackstersBarrel:fromCPs'
    )
)
