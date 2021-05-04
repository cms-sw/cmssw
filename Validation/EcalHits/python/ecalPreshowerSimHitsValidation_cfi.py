import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalPreshowerSimHitsValidation = DQMEDAnalyzer("EcalPreshowerSimHitsValidation",
    EEHitsCollection = cms.string('EcalHitsEE'),
    ESHitsCollection = cms.string('EcalHitsES'),
    moduleLabelG4 = cms.string('g4SimHits'),
    verbose = cms.untracked.bool(False),
    moduleLabelMC = cms.string('generatorSmeared')
)


