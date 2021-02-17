import FWCore.ParameterSet.Config as cms

multiClusterCaloParticleAssociation = cms.EDProducer("MCToCPAssociatorEDProducer",
    associator = cms.InputTag('mcAssocByEnergyScoreProducer'),
    label_cp = cms.InputTag("mix","MergedCaloTruth"),
    label_mcl = cms.InputTag("ticlMultiClustersFromTrackstersMerge")
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(multiClusterCaloParticleAssociation,
    label_cp = "mixData:MergedCaloTruth"
)
