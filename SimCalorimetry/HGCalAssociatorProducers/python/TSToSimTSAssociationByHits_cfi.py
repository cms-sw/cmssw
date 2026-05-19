import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HGCalAssociatorProducers.HitToTracksterAssociation_cfi import *
from SimCalorimetry.HGCalAssociatorProducers.AllTracksterToSimTracksterAssociatorsByHitsProducer_cfi import AllTracksterToSimTracksterAssociatorsByHitsProducer
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabelsPSet

allTrackstersToSimTrackstersAssociationsByHits = AllTracksterToSimTracksterAssociatorsByHitsProducer.clone(    
    tracksterCollections = cms.VInputTag(
        *[cms.InputTag(label) for label in ticlIterLabelsPSet.labels]
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
