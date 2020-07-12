import FWCore.ParameterSet.Config as cms

trackingRecHitThinningProducer = cms.EDProducer("TrackingRecHitThinningProducer",
                                                inputTag = cms.InputTag("generalTracks"),
                                                trackExtraTag = cms.InputTag("generalTracks"),
                                                )
