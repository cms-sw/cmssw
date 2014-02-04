import FWCore.ParameterSet.Config as cms



from Validation.MuonGEMDigis.simTrackMatching_cfi import SimTrackMatching
gemDigiValidation = cms.EDAnalyzer('MuonGEMDigis',
	outputFile = cms.string(''),
	stripLabel= cms.InputTag('simMuonGEMDigis'),
	cscPadLabel = cms.InputTag('simMuonGEMCSCPadDigis'),
	cscCopadLabel = cms.InputTag('simMuonGEMCSCPadDigis','Coincidence') ,
        simInputLabel = cms.untracked.string('g4SimHits'),
	minPt = cms.untracked.double(5.),
	maxEta = cms.untracked.double(2.18),
	minEta = cms.untracked.double(1.55), 
        simTrackMatching = SimTrackMatching
)
