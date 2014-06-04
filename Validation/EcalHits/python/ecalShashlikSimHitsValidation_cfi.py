import FWCore.ParameterSet.Config as cms

ecalShashlikSimHitsValidation = cms.EDAnalyzer("EcalShashlikSimHitsValidation",
    EKHitsCollection = cms.string('EcalHitsEK'),
    moduleLabelG4 = cms.string('g4SimHits'),
    ValidationCollection = cms.string('EcalValidInfo'),
    verbose = cms.untracked.bool(False)
)


