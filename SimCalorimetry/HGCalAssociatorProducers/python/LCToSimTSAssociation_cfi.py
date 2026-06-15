import FWCore.ParameterSet.Config as cms
# these do not seem to be ever used anywhere
layerClusterSimTracksterAssociation = cms.EDProducer("LCToSimTSAssociatorEDProducer",
    label_lc = cms.InputTag("hgcalMergeLayerClusters"),
    label_simTst = cms.InputTag("ticlSimTracksters", "fromLegacySimCluster"),
    associator = cms.InputTag('lcSimTSAssocByEnergyScoreProducer'),
    label_scl = cms.InputTag("mix","MergedCaloTruth"),
    associator_sc = cms.InputTag('layerClusterSimClusterAssociationProducer'),
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(layerClusterSimTracksterAssociation,
    label_cp = "mixData:MergedCaloTruth",
    label_scl = "mixData:MergedCaloTruth"
)

layerClusterSimTracksterAssociationHFNose = layerClusterSimTracksterAssociation.clone(
    label_lc = "hgcalLayerClustersHFNose"
)

layerClusterSimTracksterAssociationBarrel = layerClusterSimTracksterAssociation.clone(
    label_simTst = cms.InputTag("ticlSimTrackstersBarrel", "fromLegacySimCluster")
)
