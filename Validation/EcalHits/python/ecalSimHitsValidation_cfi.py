import FWCore.ParameterSet.Config as cms

ecalSimHitsValidation = cms.EDAnalyzer("EcalSimHitsValidation",
    outputFile = cms.untracked.string(''),
    verbose = cms.untracked.bool(False),
    moduleLabelMC = cms.string('generator'),
    EBHitsCollection = cms.string('EcalHitsEB'),
    EEHitsCollection = cms.string('EcalHitsEE'),
    EKHitsCollection = cms.string('EcalHitsEK'),
    ESHitsCollection = cms.string('EcalHitsES'),
    moduleLabelG4 = cms.string('g4SimHits')
)


