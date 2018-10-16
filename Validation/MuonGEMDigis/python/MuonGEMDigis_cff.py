import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
gemStripValidation = DQMEDAnalyzer('GEMStripDigiValidation',
  outputFile = cms.string(''),
  stripLabel= cms.InputTag('muonGEMDigis'),
  simInputLabel = cms.InputTag('g4SimHits',"MuonGEMHits"),
  # st1, st2 of xbin, st1, st2 of ybin
  nBinGlobalZR = cms.untracked.vdouble(200,200,150,250), 
  # st1 xmin xmax, st2 xmin xmax, st1 ymin ymax, st2 ymin ymax
  RangeGlobalZR = cms.untracked.vdouble(564,574,792,802,110,290,120,390), 
  nBinGlobalXY = cms.untracked.int32(360),
  detailPlot = cms.bool(False), 
)
gemPadValidation = DQMEDAnalyzer('GEMPadDigiValidation',
  outputFile = cms.string(''),
  PadLabel = cms.InputTag('simMuonGEMPadDigis'),
  simInputLabel = cms.InputTag('g4SimHits',"MuonGEMHits"),
  nBinGlobalZR = cms.untracked.vdouble(200,200,150,250), 
  RangeGlobalZR = cms.untracked.vdouble(564,574,792,802,110,290,120,390), 
  nBinGlobalXY = cms.untracked.int32(360), 
  detailPlot = cms.bool(False), 
)
gemCoPadValidation = DQMEDAnalyzer('GEMCoPadDigiValidation',
  outputFile = cms.string(''),
  CopadLabel = cms.InputTag('simCscTriggerPrimitiveDigis') ,
  simInputLabel = cms.InputTag('g4SimHits',"MuonGEMHits"),
  nBinGlobalZR = cms.untracked.vdouble(200,200,150,250), 
  RangeGlobalZR = cms.untracked.vdouble(564,574,792,802,110,290,120,390), 
  nBinGlobalXY = cms.untracked.int32(360), 
  detailPlot = cms.bool(False), 
  minBXGEM = cms.int32(-1),
  maxBXGEM = cms.int32(1),
)

gemDigiTrackValidation = DQMEDAnalyzer('GEMDigiTrackMatch',
  simInputLabel = cms.untracked.string('g4SimHits'),
  simTrackCollection = cms.InputTag('g4SimHits'),
  simVertexCollection = cms.InputTag('g4SimHits'),
  verboseSimHit = cms.untracked.int32(0),
  # GEM digi matching:
  verboseGEMDigi = cms.untracked.int32(0),
  gemDigiInput = cms.InputTag("muonGEMDigis"),
  gemPadDigiInput = cms.InputTag("simMuonGEMPadDigis"),
  gemCoPadDigiInput = cms.InputTag("simCscTriggerPrimitiveDigis"),
  minBXGEM = cms.untracked.int32(-1),
  maxBXGEM = cms.untracked.int32(1),
  matchDeltaStripGEM = cms.untracked.int32(1),
  gemMinPt = cms.untracked.double(5.0),
  gemMinEta = cms.untracked.double(1.55),
  gemMaxEta = cms.untracked.double(2.45),
  detailPlot = cms.bool(False), 
)

gemGeometryChecker = DQMEDAnalyzer('GEMCheckGeometry',
  detailPlot = cms.bool(False), 
)

gemDigiValidation = cms.Sequence( gemStripValidation+gemPadValidation+gemCoPadValidation+gemDigiTrackValidation+gemGeometryChecker)
 
from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
run2_GEM_2017.toReplaceWith(gemDigiValidation, gemDigiValidation.copyAndExclude([gemPadValidation,gemCoPadValidation]))
