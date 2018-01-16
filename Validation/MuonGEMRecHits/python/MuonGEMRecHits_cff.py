import FWCore.ParameterSet.Config as cms


gemRecHitsValidation = DQMStep1Module('GEMRecHitsValidation',
    verboseSimHit = cms.untracked.int32(1),
    simInputLabel = cms.InputTag('g4SimHits',"MuonGEMHits"),
    recHitsInputLabel = cms.InputTag('gemRecHits'),
    # st1, st2 of xbin, st1, st2 of ybin
    nBinGlobalZR = cms.untracked.vdouble(200,200,150,250), 
    # st1 xmin xmax, st2 xmin xmax, st1 ymin ymax, st2 ymin ymax
    RangeGlobalZR = cms.untracked.vdouble(564,574,792,802,110,290,120,390), 
    nBinGlobalXY = cms.untracked.int32(720), 
    detailPlot = cms.bool(False),
)

gemRecHitTrackValidation = DQMStep1Module('GEMRecHitTrackMatch',
  simInputLabel = cms.untracked.string('g4SimHits'),
  simTrackCollection = cms.InputTag('g4SimHits'),
  simVertexCollection = cms.InputTag('g4SimHits'),
  verboseSimHit = cms.untracked.int32(0),
  # GEM RecHit matching:
  verboseGEMDigi = cms.untracked.int32(0),
  gemRecHitInput = cms.InputTag("gemRecHits"),
  minBXGEM = cms.untracked.int32(-1),
  maxBXGEM = cms.untracked.int32(1),
  matchDeltaStripGEM = cms.untracked.int32(1),
  gemMinPt = cms.untracked.double(5.0),
  gemMinEta = cms.untracked.double(1.55),
  gemMaxEta = cms.untracked.double(2.45),
  detailPlot = cms.bool(False),
)

gemLocalRecoValidation = cms.Sequence( gemRecHitsValidation+gemRecHitTrackValidation )
