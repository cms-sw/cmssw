import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HGCalAssociatorProducers.LCToTSAssociator_cfi import layerClusterToCLUE3DTracksterAssociation, layerClusterToTracksterMergeAssociation, layerClusterToSimTracksterAssociation, layerClusterToSimTracksterFromCPsAssociation

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
from Configuration.ProcessModifiers.ticl_superclustering_mustache_ticl_cff import ticl_superclustering_mustache_ticl


from SimCalorimetry.HGCalAssociatorProducers.AllTracksterToSimTracksterAssociatorsByLCsProducer_cfi import AllTracksterToSimTracksterAssociatorsByLCsProducer
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabels, associatorsInstances

allTrackstersToSimTrackstersAssociationsByLCs = AllTracksterToSimTracksterAssociatorsByLCsProducer.clone(    
    tracksterCollections = cms.VInputTag(
        *[cms.InputTag(label) for label in ticlIterLabels]
    ),
    simTracksterCollections = cms.VInputTag(
      cms.InputTag('ticlSimTracksters'),
      cms.InputTag('ticlSimTracksters','fromCPs')
    ),
)
