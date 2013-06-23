import FWCore.ParameterSet.Config as cms

L1TkStubsFromSimHits = cms.EDProducer("L1TkStubBuilder_PSimHit_",
    L1TkClusters = cms.InputTag("L1TkClustersFromSimHits"),
)

L1TkStubsFromPixelDigis = cms.EDProducer("L1TkStubBuilder_PixelDigi_",
    L1TkClusters = cms.InputTag("L1TkClustersFromPixelDigis"),
)


