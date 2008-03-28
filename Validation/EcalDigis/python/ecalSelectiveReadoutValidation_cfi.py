import FWCore.ParameterSet.Config as cms

ecalSelectiveReadoutValidation = cms.EDFilter("EcalSelectiveReadoutValidation",
    TrigPrimCollection = cms.InputTag("ecalTriggerPrimitiveDigis"),
    outputFile = cms.untracked.string('EcalSelectiveReadoutValidationHistos.root'),
    verbose = cms.untracked.bool(True),
    EeDigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    EbDigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    EeRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    EeUnsuppressedDigiCollection = cms.InputTag("ecalUnsuppressedDigis"),
    EbRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    EbSimHitCollection = cms.InputTag("g4SimHits","EcalHitsEB"),
    EbUnsuppressedDigiCollection = cms.InputTag("ecalUnsuppressedDigis"),
    weights = cms.vdouble(-0.295252, -0.295252, -0.295252, -0.286034, 0.240376, 0.402839, 0.322126, 0.172504, 0.0339461, 0.0), ##my computation

    EeSrFlagCollection = cms.InputTag("ecalDigis","eeSrFlags"),
    EbSrFlagCollection = cms.InputTag("ecalDigis","ebSrFlags"),
    EeSimHitCollection = cms.InputTag("g4SimHits","EcalHitsEE"),
    LocalReco = cms.bool(True)
)


