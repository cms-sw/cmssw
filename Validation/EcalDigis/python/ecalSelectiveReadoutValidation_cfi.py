import FWCore.ParameterSet.Config as cms

ecalSelectiveReadoutValidation = cms.EDFilter("EcalSelectiveReadoutValidation",
    TrigPrimCollection = cms.InputTag("simEcalTriggerPrimitiveDigis"),
    outputFile = cms.untracked.string('EcalSelectiveReadoutValidationHistos.root'),
    verbose = cms.untracked.bool(True),
    EeDigiCollection = cms.InputTag("simEcalDigis","eeDigis"),
    EbDigiCollection = cms.InputTag("simEcalDigis","ebDigis"),
    EeRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    EeUnsuppressedDigiCollection = cms.InputTag("simEcalUnsuppressedDigis"),
    EbRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    EbSimHitCollection = cms.InputTag("g4SimHits","EcalHitsEB"),
    EbUnsuppressedDigiCollection = cms.InputTag("simEcalUnsuppressedDigis"),
    weights = cms.vdouble(-0.295252, -0.295252, -0.295252, -0.286034, 0.240376,  ##my computation

        0.402839, 0.322126, 0.172504, 0.0339461, 0.0),
    EeSrFlagCollection = cms.InputTag("simEcalDigis","eeSrFlags"),
    EbSrFlagCollection = cms.InputTag("simEcalDigis","ebSrFlags"),
    EeSimHitCollection = cms.InputTag("g4SimHits","EcalHitsEE"),
    LocalReco = cms.bool(True)
)


