import FWCore.ParameterSet.Config as cms

ecalEndcapSimHitsValidation = cms.EDAnalyzer("EcalEndcapSimHitsValidation",
    EEHitsCollection = cms.string('EcalHitsEE'),
    moduleLabelG4 = cms.string('g4SimHits'),
    ValidationCollection = cms.string('EcalValidInfo'),
    verbose = cms.untracked.bool(False)
)


