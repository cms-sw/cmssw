import FWCore.ParameterSet.Config as cms




gemDigiValidation = cms.EDAnalyzer('MuonGEMDigis',
	outputFile = cms.string('valid.root'),
	stripLabel= cms.InputTag('simMuonGEMDigis'),
	cscPadLabel = cms.InputTag('simMuonGEMCSCPadDigis'),
	cscCopadLabel = cms.InputTag('simMuonGEMCSCPadDigis','Coincidence') )
