import FWCore.ParameterSet.Config as cms

onlineEmbeddingBeamSpotProducer = cms.EDProducer('EmbeddingBeamSpotOnlineProducer',
                                        src = cms.InputTag('offlineBeamSpot'),
)

