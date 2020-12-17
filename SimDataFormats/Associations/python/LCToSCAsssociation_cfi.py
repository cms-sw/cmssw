import FWCore.ParameterSet.Config as cms

layerClusterSimClusterAsssociation = cms.EDProducer("LCToSCAssociatorEDProducer",
    associator = cms.InputTag('scAssocByEnergyScoreProducer'),
    label_scl = cms.InputTag("mix","MergedCaloTruth"),
    label_lcl = cms.InputTag("hgcalLayerClusters")
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(layerClusterSimClusterAsssociation,
    label_scl = "mixData:MergedCaloTruth"
)
