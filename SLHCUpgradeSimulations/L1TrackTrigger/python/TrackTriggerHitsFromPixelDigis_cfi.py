import FWCore.ParameterSet.Config as cms

TrackTriggerHitsFromPixelDigis = cms.EDProducer("TrackTriggerHitProducer",
    threshold = cms.uint32(30),
    input = cms.InputTag("simSiPixelDigis")
)




