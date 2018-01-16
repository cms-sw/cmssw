import FWCore.ParameterSet.Config as cms


from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
me0DigiValidation = DQMEDAnalyzer('ME0DigisValidation',
    verboseSimHit = cms.untracked.int32(1),
    simInputLabel = cms.InputTag('g4SimHits',"MuonME0Hits"),
    digiInputLabel = cms.InputTag("simMuonME0PseudoReDigis"),
    sigma_x = cms.double(0.0003), #It corresponds to phi resolution
    sigma_y = cms.double(0.03), #It corresponds to eta resolution
    # st1, st2_short, st2_long of xbin, st1,st2_short,st2_long of ybin
    nBinGlobalZR = cms.untracked.vdouble(30,100),
    # st1 xmin, xmax, st2_short xmin, xmax, st2_long xmin, xmax, st1 ymin, ymax...
    RangeGlobalZR = cms.untracked.vdouble(525,555,60,160),
    nBinGlobalXY = cms.untracked.int32(160),
)
