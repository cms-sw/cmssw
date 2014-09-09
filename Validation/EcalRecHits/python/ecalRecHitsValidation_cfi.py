import FWCore.ParameterSet.Config as cms

ecalRecHitsValidation = cms.EDAnalyzer("EcalRecHitsValidation",
    hitsProducer = cms.string('g4SimHits'),
    outputFile = cms.untracked.string(''),
    EBuncalibrechitCollection = cms.InputTag("ecalMultiFitUncalibRecHit","EcalUncalibRecHitsEB"),
    EEuncalibrechitCollection = cms.InputTag("ecalMultiFitUncalibRecHit","EcalUncalibRecHitsEE"),
    EKuncalibrechitCollection = cms.InputTag("ecalMultiFitUncalibRecHit","EcalUncalibRecHitsEK"),
    verbose = cms.untracked.bool(False),
    EBrechitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    EErechitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    EKrechitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK"),
    ESrechitCollection = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),

    
    moduleLabelMC = cms.string('generator')
)


