import FWCore.ParameterSet.Config as cms

TTClusterAssociatorFromPixelDigis = cms.EDProducer("TTClusterAssociator_PixelDigi_",
    TTClusters = cms.VInputTag( cms.InputTag("TTClustersFromPixelDigis", "ClusterInclusive"),
                                cms.InputTag("TTStubsFromPixelDigis", "ClusterAccepted"),
    ),
    simTrackHits = cms.InputTag("g4SimHits"),
)

