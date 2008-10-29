import FWCore.ParameterSet.Config as cms

ecalPreshowerRecHitsValidation = cms.EDFilter("EcalPreshowerRecHitsValidation",
    EErechitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    ESrechitCollection = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    EEuncalibrechitCollection = cms.InputTag("ecalWeightUncalibRecHit","EcalUncalibRecHitsEE"),
    verbose = cms.untracked.bool(False)
)



