import FWCore.ParameterSet.Config as cms

TTClusterAssociatorFromPixelDigis = cms.EDProducer("TTClusterAssociator_PixelDigi_",
    TTClusters = cms.InputTag("TTClustersFromPixelDigis"),
    simTrackHits = cms.InputTag("g4SimHits"),
)

