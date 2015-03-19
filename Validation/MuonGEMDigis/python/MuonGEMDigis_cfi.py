import FWCore.ParameterSet.Config as cms

gemStripValidation = cms.EDAnalyzer('GEMStripDigiValidation',
  outputFile = cms.string(''),
  stripLabel= cms.InputTag('simMuonGEMDigis'),
  simInputLabel = cms.InputTag('g4SimHits',"MuonGEMHits"),
  nBinGlobalZR = cms.untracked.vdouble(200,200,200,150,180,250), 
  RangeGlobalZR = cms.untracked.vdouble(564,572,786,794,794,802,110,260,170,350,100,350), 
  nBinGlobalXY = cms.untracked.int32(360), 
)
gemPadValidation = cms.EDAnalyzer('GEMPadDigiValidation',
  outputFile = cms.string(''),
  PadLabel = cms.InputTag('simMuonGEMPadDigis'),
  simInputLabel = cms.InputTag('g4SimHits',"MuonGEMHits"),
  nBinGlobalZR = cms.untracked.vdouble(200,200,200,150,180,250), 
  RangeGlobalZR = cms.untracked.vdouble(564,572,786,794,794,802,110,260,170,350,100,350), 
  nBinGlobalXY = cms.untracked.int32(360), 
)
gemCoPadValidation = cms.EDAnalyzer('GEMCoPadDigiValidation',
  outputFile = cms.string(''),
  CopadLabel = cms.InputTag('simMuonGEMPadDigis') ,
  simInputLabel = cms.InputTag('g4SimHits',"MuonGEMHits"),
  nBinGlobalZR = cms.untracked.vdouble(200,200,200,150,180,250), 
  RangeGlobalZR = cms.untracked.vdouble(564,572,786,794,794,802,110,260,170,350,100,350), 
  nBinGlobalXY = cms.untracked.int32(360), 
)


#gemDigiValidation = cms.Sequence( gemStripValidation+gemPadValidation+gemCoPadValidation)
gemDigiValidation = cms.Sequence( gemStripValidation+gemPadValidation)
