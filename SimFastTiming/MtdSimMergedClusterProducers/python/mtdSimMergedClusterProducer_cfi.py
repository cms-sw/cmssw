import FWCore.ParameterSet.Config as cms

mtdSimMergedClusterProducer = cms.EDProducer(
    "MtdSimMergedClusterProducer",
    trackingParticles = cms.InputTag("mix", "MergedTrackTruth"),
    mtdSimLayerClusters = cms.InputTag("mix", "MergedMtdTruthLC"),
    tp2SimAssociationMap = cms.InputTag("mtdSimLayerClusterToTPAssociation"),
    minClusterEnergy = cms.double(0)  # GeV
)