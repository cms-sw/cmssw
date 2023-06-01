import FWCore.ParameterSet.Config as cms

layerClusterSimClusterAssociation = cms.EDProducer("LCToSCAssociatorEDProducer",
    associator = cms.InputTag('scAssocByEnergyScoreProducer'),
    label_scl = cms.InputTag("mix","MergedCaloTruth"),
    label_lcl = cms.InputTag("hgcalMergeLayerClusters")
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(layerClusterSimClusterAssociation,
    label_scl = "mixData:MergedCaloTruth"
)

layerClusterSimClusterAssociationHFNose = layerClusterSimClusterAssociation.clone(
    label_lcl = "hgcalLayerClustersHFNose"
)
