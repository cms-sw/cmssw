import FWCore.ParameterSet.Config as cms

gemStripValidation = cms.EDAnalyzer('GEMStripDigiValidation',
  outputFile = cms.string(''),
  stripLabel= cms.InputTag('simMuonGEMDigis'),
  simInputLabel = cms.InputTag('g4SimHits',"MuonGEMHits"),
  nBinGlobalZR = cms.untracked.vdouble(200,200,200,150,180,250), 
  RangeGlobalZR = cms.untracked.vdouble(564,572,786,794,786,802,110,260,170,350,100,350), 
  nBinGlobalXY = cms.untracked.int32(360),
  detailPlot = cms.bool(False), 
)
gemPadValidation = cms.EDAnalyzer('GEMPadDigiValidation',
  outputFile = cms.string(''),
  PadLabel = cms.InputTag('simMuonGEMPadDigis'),
  simInputLabel = cms.InputTag('g4SimHits',"MuonGEMHits"),
  nBinGlobalZR = cms.untracked.vdouble(200,200,200,150,180,250), 
  RangeGlobalZR = cms.untracked.vdouble(564,572,786,794,786,802,110,260,170,350,100,350), 
  nBinGlobalXY = cms.untracked.int32(360), 
  detailPlot = cms.bool(False), 
)
gemCoPadValidation = cms.EDAnalyzer('GEMCoPadDigiValidation',
  outputFile = cms.string(''),
  CopadLabel = cms.InputTag('simCscTriggerPrimitiveDigis') ,
  simInputLabel = cms.InputTag('g4SimHits',"MuonGEMHits"),
  nBinGlobalZR = cms.untracked.vdouble(200,200,200,150,180,250), 
  RangeGlobalZR = cms.untracked.vdouble(564,572,786,794,786,802,110,260,170,350,100,350), 
  nBinGlobalXY = cms.untracked.int32(360), 
  detailPlot = cms.bool(False), 
  minBXGEM = cms.int32(-1),
  maxBXGEM = cms.int32(1),
)

gemDigiTrackValidation = cms.EDAnalyzer('GEMDigiTrackMatch',
  simInputLabel = cms.untracked.string('g4SimHits'),
  simTrackCollection = cms.InputTag('g4SimHits'),
  simVertexCollection = cms.InputTag('g4SimHits'),
  verboseSimHit = cms.untracked.int32(0),
  # GEM digi matching:
  verboseGEMDigi = cms.untracked.int32(0),
  gemDigiInput = cms.InputTag("simMuonGEMDigis"),
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

gemGeometryChecker = cms.EDAnalyzer('GEMCheckGeometry',
  detailPlot = cms.bool(False), 
)

gemDigiValidation = cms.Sequence( gemStripValidation+gemPadValidation+gemCoPadValidation+gemDigiTrackValidation+gemGeometryChecker)
