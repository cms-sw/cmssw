import FWCore.ParameterSet.Config as cms


me0RecHitValidation = cms.EDAnalyzer('ME0RecHitsValidation',
    verboseSimHit = cms.untracked.int32(1),
    simInputLabel = cms.InputTag('g4SimHits',"MuonME0Hits"),
    recHitInputLabel = cms.InputTag("me0RecHits"),
    segmentInputLabel = cms.InputTag("me0Segment"),
    # st1, st2_short, st2_long of xbin, st1,st2_short,st2_long of ybin
    nBinGlobalZR = cms.untracked.vdouble(80,120),
    # st1 xmin, xmax, st2_short xmin, xmax, st2_long xmin, xmax, st1 ymin, ymax...
    RangeGlobalZR = cms.untracked.vdouble(515,555,20,160),
    nBinGlobalXY = cms.untracked.int32(360),
)

me0SegmentsValidation = cms.EDAnalyzer('ME0SegmentsValidation',
    verboseSimHit = cms.untracked.int32(1),
    segmentInputLabel = cms.InputTag("me0Segment"),
    # st1, st2_short, st2_long of xbin, st1,st2_short,st2_long of ybin
    nBinGlobalZR = cms.untracked.vdouble(80,120),
    # st1 xmin, xmax, st2_short xmin, xmax, st2_long xmin, xmax, st1 ymin, ymax...
    RangeGlobalZR = cms.untracked.vdouble(515,555,20,160),
    nBinGlobalXY = cms.untracked.int32(360),
)


me0LocalRecoValidation = cms.sequence( me0RecHitValidation + me0SegmentsValidation )
