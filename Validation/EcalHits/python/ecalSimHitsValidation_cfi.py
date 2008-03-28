import FWCore.ParameterSet.Config as cms

ecalSimHitsValidation = cms.EDFilter("EcalSimHitsValidation",
    ESHitsCollection = cms.string('EcalHitsES'),
    outputFile = cms.untracked.string('EcalSimHitsValidation.root'),
    verbose = cms.untracked.bool(True),
    moduleLabelMC = cms.string('source'),
    EBHitsCollection = cms.string('EcalHitsEB'),
    EEHitsCollection = cms.string('EcalHitsEE'),
    moduleLabelG4 = cms.string('g4SimHits')
)


