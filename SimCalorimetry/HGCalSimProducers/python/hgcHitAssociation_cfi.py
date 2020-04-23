import FWCore.ParameterSet.Config as cms

HGCRecHitMapProducer = cms.EDProducer("HGCalRecHitMapProducer",
)

LCAssocByEnergyScoreProducer = cms.EDProducer("LayerClusterAssociatorByEnergyScoreProducer",
)
