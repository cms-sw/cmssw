import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HGCalAssociatorProducers.HitToTracksterAssociation_cfi import *
from SimCalorimetry.HGCalAssociatorProducers.AllTracksterToSimTracksterAssociatorsByHitsProducer_cfi import AllTracksterToSimTracksterAssociatorsByHitsProducer as _AllTracksterToSimTracksterAssociatorsByHitsProducer
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabelsPSet

allTrackstersToSimTrackstersAssociationsByHits = _AllTracksterToSimTracksterAssociatorsByHitsProducer.clone(    
    tracksterCollections = cms.VInputTag(
        *[cms.InputTag(label) for label in ticlIterLabelsPSet.labels]
    ),
    simTracksters = cms.VPSet(
        cms.PSet(
            simTracksterCollection=cms.InputTag("ticlSimTracksters", "fromBoundarySimCluster"),
            hitToSimClusterMap=cms.InputTag("hitToBoundarySimClusterAssociator")
        ),
        cms.PSet(
            simTracksterCollection=cms.InputTag("ticlSimTracksters", "fromCaloParticle"),
            hitToSimClusterMap=cms.InputTag("hitToCPSimClusterAssociator")
        ),
    )
)


## Barrel
from SimCalorimetry.HGCalAssociatorProducers.AllBarrelTracksterToSimTracksterAssociatorsByHitsProducer_cfi import AllBarrelTracksterToSimTracksterAssociatorsByHitsProducer

allBarrelTrackstersToSimTrackstersAssociationsByHits = AllBarrelTracksterToSimTracksterAssociatorsByHitsProducer.clone(
    allHitToTSAccoc = cms.string('allHitToBarrelTracksterAssociations'),
    tracksterCollections = cms.VInputTag('ticlTrackstersCLUE3DBarrel'),
    simTracksters = cms.VPSet(
        cms.PSet(
            simTracksterCollection=cms.InputTag("ticlSimTrackstersBarrel", "fromBoundarySimCluster"),
            hitToSimClusterMap=cms.InputTag("barrelHitToBoundarySimClusterAssociator")
        ),
        cms.PSet(
            simTracksterCollection=cms.InputTag("ticlSimTrackstersBarrel", "fromCaloParticle"),
            hitToSimClusterMap=cms.InputTag("barrelHitToCPSimClusterAssociator")
        ),
    )
)
