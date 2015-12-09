import FWCore.ParameterSet.Config as cms


gemSimHitValidation = cms.EDAnalyzer('GEMHitsValidation',
    verboseSimHit = cms.untracked.int32(1),
    simInputLabel = cms.InputTag('g4SimHits',"MuonGEMHits"),
    # st1, st2_short, st2_long of xbin, st1,st2_short,st2_long of ybin
    nBinGlobalZR = cms.untracked.vdouble(200,200,200,110,140,210), 
    # st1 xmin, xmax, st2_short xmin, xmax, st2_long xmin, xmax, st1 ymin, ymax...
    RangeGlobalZR = cms.untracked.vdouble(564,572,786,794,794,802,130,240,190,330,120,330),
    nBinGlobalXY = cms.untracked.int32(720),
    detailPlot = cms.bool(False), 
)

gemSimTrackValidation = cms.EDAnalyzer('GEMSimTrackMatch',
    verboseSimHit = cms.untracked.int32(1),
    simInputLabel = cms.untracked.string('g4SimHits'),
    simMuOnlyGEM = cms.untracked.bool(True),
    discardEleHitsGEM = cms.untracked.bool(True),
    simTrackCollection = cms.InputTag('g4SimHits'),
    simVertexCollection = cms.InputTag('g4SimHits'),
    gemMinPt = cms.untracked.double(5.0),
    gemMinEta = cms.untracked.double(1.55),
    gemMaxEta = cms.untracked.double(2.45),
    detailPlot = cms.bool(False), 
)

gemSimValidation = cms.Sequence( gemSimHitValidation+gemSimTrackValidation)
