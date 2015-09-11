import FWCore.ParameterSet.Config as cms

ecalSimHitsValidation = cms.EDAnalyzer("EcalSimHitsValidation",
    ESHitsCollection = cms.string('EcalHitsES'),
    outputFile = cms.untracked.string(''),
    verbose = cms.untracked.bool(False),
    moduleLabelMC = cms.string('generatorSmeared'),
    EBHitsCollection = cms.string('EcalHitsEB'),
    EEHitsCollection = cms.string('EcalHitsEE'),
    moduleLabelG4 = cms.string('g4SimHits')
)


