import FWCore.ParameterSet.Config as cms

tracksterSimClusterAssociation = cms.EDProducer("TSToSCAssociatorEDProducer",
    associator = cms.InputTag('tsAssocByEnergyScoreProducer'),
    label_scl = cms.InputTag("mix","MergedCaloTruth"),
    label_tst = cms.InputTag("ticlTrackstersMerge"),
    label_lcl = cms.InputTag("hgcalMergeLayerClusters")
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(tracksterSimClusterAssociation,
    label_scl = "mixData:MergedCaloTruth"
)
