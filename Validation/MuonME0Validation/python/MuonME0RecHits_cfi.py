import FWCore.ParameterSet.Config as cms


from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
me0RecHitsValidation = DQMEDAnalyzer('ME0RecHitsValidation',
    verboseSimHit = cms.untracked.int32(1),
    simInputLabel = cms.InputTag('g4SimHits',"MuonME0Hits"),
    recHitInputLabel = cms.InputTag("me0RecHits"),
    segmentInputLabel = cms.InputTag("me0Segments"),
    # st1, st2_short, st2_long of xbin, st1,st2_short,st2_long of ybin
    nBinGlobalZR = cms.untracked.vdouble(30,100),
    # st1 xmin, xmax, st2_short xmin, xmax, st2_long xmin, xmax, st1 ymin, ymax...
    RangeGlobalZR = cms.untracked.vdouble(525,555,60,160),
    nBinGlobalXY = cms.untracked.int32(160),
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
me0SegmentsValidation = DQMEDAnalyzer('ME0SegmentsValidation',
    verboseSimHit = cms.untracked.int32(1),
    segmentInputLabel = cms.InputTag("me0Segments"),
    digiInputLabel = cms.InputTag("simMuonME0PseudoReDigis"),
    simInputLabel = cms.InputTag('g4SimHits',"MuonME0Hits"),
    simInputLabelST = cms.InputTag('g4SimHits'),
    sigma_x = cms.double(0.0003), #It corresponds to phi resolution
    sigma_y = cms.double(0.03), #It corresponds to eta resolution
    eta_max = cms.double(2.8),
    eta_min = cms.double(2.0),
    pt_min = cms.double(0.0),
    isMuonGun = cms.bool(True),
    # st1, st2_short, st2_long of xbin, st1,st2_short,st2_long of ybin
    nBinGlobalZR = cms.untracked.vdouble(30,100),
    # st1 xmin, xmax, st2_short xmin, xmax, st2_long xmin, xmax, st1 ymin, ymax...
    RangeGlobalZR = cms.untracked.vdouble(525,555,60,160),
    nBinGlobalXY = cms.untracked.int32(160),
)


me0LocalRecoValidation = cms.Sequence( me0RecHitsValidation + me0SegmentsValidation )
