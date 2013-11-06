import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMEnvironment_cfi import *

from DQMServices.Examples.test.ConverterTester_cfi import *


DQMStore = cms.Service("DQMStore")

dqmSaver.convention = 'Offline'
dqmSaver.workflow = '/MuonGEMDigisV/Workflow/DIGI'

from Validation.MuonGEMDigis.simTrackMatching_cfi import SimTrackMatching

gemDigiValidation = cms.EDAnalyzer('MuonGEMDigis',
	outputFile = cms.string('valid.root'),
	stripLabel= cms.InputTag('simMuonGEMDigis'),
	cscPadLabel = cms.InputTag('simMuonGEMCSCPadDigis'),
	cscCopadLabel = cms.InputTag('simMuonGEMCSCPadDigis','Coincidence') ,
        simInputLabel = cms.untracked.string('g4SimHits'),
        simTrackMatching = SimTrackMatching
)
