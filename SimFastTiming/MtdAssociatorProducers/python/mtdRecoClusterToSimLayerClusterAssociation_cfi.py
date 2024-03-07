import FWCore.ParameterSet.Config as cms

mtdRecoClusterToSimLayerClusterAssociation = cms.EDProducer("MtdRecoClusterToSimLayerClusterAssociatorEDProducer",
    associator = cms.InputTag('mtdRecoClusterToSimLayerClusterAssociatorByHits'),
    mtdSimClustersTag = cms.InputTag('mix','MergedMtdTruthLC'),
    btlRecoClustersTag = cms.InputTag('mtdClusters', 'FTLBarrel'),
    etlRecoClustersTag = cms.InputTag('mtdClusters', 'FTLEndcap'),
)
