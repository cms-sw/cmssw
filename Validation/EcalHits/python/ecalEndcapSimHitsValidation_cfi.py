import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalEndcapSimHitsValidation = DQMEDAnalyzer("EcalEndcapSimHitsValidation",
    EEHitsCollection = cms.string('EcalHitsEE'),
    moduleLabelG4 = cms.string('g4SimHits'),
    ValidationCollection = cms.string('EcalValidInfo'),
    verbose = cms.untracked.bool(False)
)


