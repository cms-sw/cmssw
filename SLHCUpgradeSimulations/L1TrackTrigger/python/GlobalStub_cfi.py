import FWCore.ParameterSet.Config as cms

GlobalStubsFromSimHits = cms.EDFilter("GlobalStubBuilder_PSimHit_",
    LocalStubs = cms.InputTag("LocalStubsFromSimHits")
)

GlobalStubsFromPixelDigis = cms.EDFilter("GlobalStubBuilder_PixelDigi_",
    LocalStubs = cms.InputTag("LocalStubsFromPixelDigis")
)

GlobalStubsFromTrackTriggerHits = cms.EDFilter("GlobalStubBuilder_TTHit_",
    LocalStubs = cms.InputTag("LocalStubsFromTrackTriggerHits")
)


