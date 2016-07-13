import FWCore.ParameterSet.Config as cms


me0HitsValidation = cms.EDAnalyzer('ME0HitsValidation',
    verboseSimHit = cms.untracked.int32(1),
    simInputLabel = cms.InputTag('g4SimHits',"MuonME0Hits"),
    # st1, st2_short, st2_long of xbin, st1,st2_short,st2_long of ybin
    nBinGlobalZR = cms.untracked.vdouble(80,120),
    # st1 xmin, xmax, st2_short xmin, xmax, st2_long xmin, xmax, st1 ymin, ymax...
    RangeGlobalZR = cms.untracked.vdouble(520,555,20,160),
    nBinGlobalXY = cms.untracked.int32(360),
)
