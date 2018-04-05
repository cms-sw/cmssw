import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalDigisValidation = DQMEDAnalyzer('EcalDigisValidation',
    outputFile = cms.untracked.string(''),
    verbose = cms.untracked.bool(False),
    EBdigiCollection = cms.InputTag("simEcalDigis","ebDigis"),
    ESdigiCollection = cms.InputTag("simEcalPreshowerDigis"),
    moduleLabelMC = cms.string('generatorSmeared'),
    EEdigiCollection = cms.InputTag("simEcalDigis","eeDigis"),
    moduleLabelG4 = cms.string('g4SimHits')
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(ecalDigisValidation, moduleLabelG4 = 'fastSimProducer')
