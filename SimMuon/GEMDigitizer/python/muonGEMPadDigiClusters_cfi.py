import FWCore.ParameterSet.Config as cms

# Module to create GEM pad digi clusters
simMuonGEMPadDigiClusters = cms.EDProducer("GEMPadDigiClusterProducer",
    InputCollection = cms.InputTag('simMuonGEMPadDigis'),
    maxClusters = cms.uint32(8),
    maxClusterSize = cms.uint32(8)
)
