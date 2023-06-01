import FWCore.ParameterSet.Config as cms

layerClusterCaloParticleAssociation = cms.EDProducer("LCToCPAssociatorEDProducer",
    associator = cms.InputTag('lcAssocByEnergyScoreProducer'),
    label_cp = cms.InputTag("mix","MergedCaloTruth"),
    label_lc = cms.InputTag("hgcalMergeLayerClusters")
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(layerClusterCaloParticleAssociation,
    label_cp = "mixData:MergedCaloTruth"
)

layerClusterCaloParticleAssociationHFNose = layerClusterCaloParticleAssociation.clone(
    label_lc = "hgcalLayerClustersHFNose"
)
