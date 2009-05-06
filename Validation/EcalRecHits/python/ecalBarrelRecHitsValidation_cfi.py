import FWCore.ParameterSet.Config as cms

ecalBarrelRecHitsValidation = cms.EDFilter("EcalBarrelRecHitsValidation",
    EBdigiCollection = cms.InputTag("simEcalDigis","ebDigis"),
    EBuncalibrechitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB"),
    verbose = cms.untracked.bool(False)
)



