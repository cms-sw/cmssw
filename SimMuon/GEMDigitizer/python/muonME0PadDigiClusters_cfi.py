import FWCore.ParameterSet.Config as cms

# Module to create ME0 pad digi clusters
simMuonME0PadDigiClusters = cms.EDProducer("ME0PadDigiClusterProducer",
    InputCollection = cms.InputTag('simMuonME0PadDigis'),
    maxClusters = cms.uint32(24),
    maxClusterSize = cms.uint32(8)
)
