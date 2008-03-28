import FWCore.ParameterSet.Config as cms

ecalRecHitsValidation = cms.EDFilter("EcalRecHitsValidation",
    outputFile = cms.untracked.string('EcalRecHitsValidation.root'),
    EEuncalibrechitCollection = cms.InputTag("ecalWeightUncalibRecHit","EcalUncalibRecHitsEE"),
    verbose = cms.untracked.bool(True),
    EErechitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    ESrechitCollection = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    EBuncalibrechitCollection = cms.InputTag("ecalWeightUncalibRecHit","EcalUncalibRecHitsEB"),
    EBrechitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    moduleLabelMC = cms.string('source')
)


