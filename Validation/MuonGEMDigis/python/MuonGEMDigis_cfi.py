import FWCore.ParameterSet.Config as cms



from Validation.MuonGEMDigis.simTrackMatching_cfi import SimTrackMatching
gemDigiValidation = cms.EDAnalyzer('MuonGEMDigis',
	outputFile = cms.string('valid.root'),
	stripLabel= cms.InputTag('simMuonGEMDigis'),
	cscPadLabel = cms.InputTag('simMuonGEMCSCPadDigis'),
	cscCopadLabel = cms.InputTag('simMuonGEMCSCPadDigis','Coincidence') ,
        simInputLabel = cms.untracked.string('g4SimHits'),
        simTrackMatching = SimTrackMatching
)
