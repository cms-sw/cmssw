import FWCore.ParameterSet.Config as cms

ecalBarrelSimHitsValidation = cms.EDAnalyzer("EcalBarrelSimHitsValidation",
    moduleLabelG4 = cms.string('g4SimHits'),
    verbose = cms.untracked.bool(False),
    ValidationCollection = cms.string('EcalValidInfo'),
    EBHitsCollection = cms.string('EcalHitsEB')
)


