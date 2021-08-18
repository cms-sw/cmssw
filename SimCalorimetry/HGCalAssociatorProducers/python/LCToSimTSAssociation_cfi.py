import FWCore.ParameterSet.Config as cms

layerClusterSimTracksterAssociation = cms.EDProducer("LCToSimTSAssociatorEDProducer",
    label_lc = cms.InputTag("hgcalLayerClusters"),
    label_simTst = cms.InputTag("ticlSimTracksters"),
    associator = cms.InputTag('lcSimTSAssocByEnergyScoreProducer'),
    label_cp = cms.InputTag("mix","MergedCaloTruth"),
    associator_cp = cms.InputTag('layerClusterCaloParticleAssociationProducer'),
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
