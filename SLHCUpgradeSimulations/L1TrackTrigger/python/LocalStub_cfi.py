import FWCore.ParameterSet.Config as cms

LocalStubsFromSimHits = cms.EDFilter("LocalStubBuilder_PSimHit_",
    rawHits = cms.VInputTag(cms.InputTag("famosSimHits","TrackerHits")),
    storeClusters = cms.bool(False)
)

LocalStubsFromPixelDigis = cms.EDFilter("LocalStubBuilder_PixelDigi_",
    rawHits = cms.VInputTag(cms.InputTag("simSiPixelDigis")),
    ADCThreshold = cms.uint32(30),
    storeClusters = cms.bool(False)
)

LocalStubsFromTrackTriggerHits = cms.EDFilter("LocalStubBuilder_TTHit_",
    rawHits = cms.VInputTag(cms.InputTag("TrackTriggerHitsFromPixelDigis")),
    storeClusters = cms.bool(False)
)


