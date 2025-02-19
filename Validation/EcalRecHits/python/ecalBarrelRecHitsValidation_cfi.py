import FWCore.ParameterSet.Config as cms

ecalBarrelRecHitsValidation = cms.EDAnalyzer("EcalBarrelRecHitsValidation",
    EBdigiCollection = cms.InputTag("simEcalDigis","ebDigis"),
    EBuncalibrechitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB"),
    verbose = cms.untracked.bool(False)
)



