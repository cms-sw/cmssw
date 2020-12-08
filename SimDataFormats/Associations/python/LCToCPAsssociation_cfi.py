import FWCore.ParameterSet.Config as cms

trackingParticleRecoTrackAsssociation = cms.EDProducer("LCToCPAssociatorEDProducer",
    associator = cms.InputTag('lcAssocByEnergyScoreProducer'),
    label_cp = cms.InputTag("mix","MergedCaloTruth"),
    label_lc = cms.InputTag("hgcalLayerClusters")
)
