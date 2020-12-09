import FWCore.ParameterSet.Config as cms

trackingParticleRecoTrackAsssociation = cms.EDProducer("LCToCPAssociatorEDProducer",
    associator = cms.InputTag('lcAssocByEnergyScoreProducer'),
    label_cp = cms.InputTag("mix","MergedCaloTruth"),
    label_lc = cms.InputTag("hgcalLayerClusters")
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(trackingParticleRecoTrackAsssociation,
    label_cp = "mixData:MergedCaloTruth"
)
