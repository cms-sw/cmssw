import FWCore.ParameterSet.Config as cms

ecalRecHitsValidation = cms.EDAnalyzer("EcalRecHitsValidation",
    hitsProducer = cms.string('g4SimHits'),
    outputFile = cms.untracked.string(''),
    EEuncalibrechitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE"),
    verbose = cms.untracked.bool(False),
    EErechitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    ESrechitCollection = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    EBuncalibrechitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB"),
    EBrechitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    moduleLabelMC = cms.string('generator')
)


