import FWCore.ParameterSet.Config as cms

ecalDigisValidation = cms.EDAnalyzer("EcalDigisValidation",
    outputFile = cms.untracked.string(''),
    verbose = cms.untracked.bool(False),
    EBdigiCollection = cms.InputTag("simEcalDigis","ebDigis"),
    ESdigiCollection = cms.InputTag("simEcalPreshowerDigis"),
    moduleLabelMC = cms.string('generatorSmeared'),
    EEdigiCollection = cms.InputTag("simEcalDigis","eeDigis"),
    moduleLabelG4 = cms.string('g4SimHits')
)

from Configuration.StandardSequences.Eras import eras
if eras.fastSim.isChosen():
    ecalDigisValidation.moduleLabelG4 = cms.string('famosSimHits')


