import FWCore.ParameterSet.Config as cms

# Clusterizer options
siPhase2Clusters = cms.EDProducer('SimTrackerSiPhase2Clusterizer',
    src = cms.InputTag("simSiPixelDigis"), 
    maxClusterSize = cms.int32(8),
    maxNumberClusters = cms.int32(-1),
    clusterSimLink = cms.bool(True)
)



