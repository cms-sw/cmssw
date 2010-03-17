import FWCore.ParameterSet.Config as cms

TrackletsFromSimHits = cms.EDFilter("TrackletBuilder_PSimHit_",
    minPtThreshold = cms.double(2.0),
    ipWidth = cms.double(15.0),
	fastPhiCut = cms.double(0.1),
    GlobalStubs = cms.InputTag("GlobalStubsFromSimHits")
)

TrackletsFromPixelDigis = cms.EDFilter("TrackletBuilder_PixelDigi_",
    minPtThreshold = cms.double(2.0),
    ipWidth = cms.double(15.0),
	fastPhiCut = cms.double(0.1),
    GlobalStubs = cms.InputTag("GlobalStubsFromPixelDigis")
)

TrackletsFromTrackTriggerHits = cms.EDFilter("TrackletBuilder_TTHit_",
    minPtThreshold = cms.double(2.0),
    ipWidth = cms.double(15.0),
	fastPhiCut = cms.double(0.1),
    GlobalStubs = cms.InputTag("GlobalStubsFromTrackTriggerHits")
)

