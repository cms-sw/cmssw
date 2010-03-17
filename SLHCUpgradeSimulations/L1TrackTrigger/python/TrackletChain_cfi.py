import FWCore.ParameterSet.Config as cms

TrackletChainsFromSimHits = cms.EDFilter("TrackletChainBuilder_PSimHit_",
    minPtThreshold = cms.double(5.0),
    Zmatch = cms.double(1.0),
    ShortTracklets = cms.InputTag("TrackletsFromSimHits", "ShortTracklets" )
)

TrackletChainsFromPixelDigis = cms.EDFilter("TrackletChainBuilder_PixelDigi_",
    minPtThreshold = cms.double(5.0),
    Zmatch = cms.double(1.0),
    ShortTracklets = cms.InputTag("TrackletsFromPixelDigis", "ShortTracklets" )
)

TrackletChainsFromTrackTriggerHits = cms.EDFilter("TrackletChainBuilder_TTHit_",
    minPtThreshold = cms.double(5.0),
    Zmatch = cms.double(1.0),
    ShortTracklets = cms.InputTag("TrackletsFromTrackTriggerHits", "ShortTracklets" )
)

