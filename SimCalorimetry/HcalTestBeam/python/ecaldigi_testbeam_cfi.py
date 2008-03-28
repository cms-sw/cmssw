import FWCore.ParameterSet.Config as cms

ecalUnsuppressedDigis = cms.EDProducer("EcalTBDigiProducer",
    photoelectronsToAnalogEndcap = cms.double(0.000555555),
    readoutFrameSize = cms.int32(10),
    timePhase = cms.double(56.1),
    ConstantTerm = cms.double(0.003),
    applyConstantTerm = cms.bool(True),
    binOfMaximum = cms.int32(6),
    simHitToPhotoelectronsEndcap = cms.double(1800.0),
    samplingFactor = cms.double(1.0),
    EcalTBInfoLabel = cms.untracked.string('simCaloTB'),
    tdcMin = cms.vint32(430, 400, 400, 400, 400),
    doNoise = cms.bool(True),
    tdcMax = cms.vint32(958, 927, 927, 927, 927),
    tunePhaseShift = cms.double(0.5),
    #  vdouble CorrelatedNoiseMatrix = { 1.00, 0.67, 0.53, 0.44, 0.39, 0.36, 0.38, 0.35, 0.36, 0.32,
    #                                    0.67, 1.00, 0.67, 0.53, 0.44, 0.39, 0.36, 0.38, 0.35, 0.36,
    #                                    0.53, 0.67, 1.00, 0.67, 0.53, 0.44, 0.39, 0.36, 0.38, 0.35,
    #                                    0.44, 0.53, 0.67, 1.00, 0.67, 0.53, 0.44, 0.39, 0.36, 0.38,
    #                                    0.39, 0.44, 0.53, 0.67, 1.00, 0.67, 0.53, 0.44, 0.39, 0.36,
    #                                    0.36, 0.39, 0.44, 0.53, 0.67, 1.00, 0.67, 0.53, 0.44, 0.39,
    #                                    0.38, 0.36, 0.39, 0.44, 0.53, 0.67, 1.00, 0.67, 0.53, 0.44,
    #                                    0.35, 0.38, 0.36, 0.39, 0.44, 0.53, 0.67, 1.00, 0.67, 0.53,
    #                                    0.36, 0.35, 0.38, 0.36, 0.39, 0.44, 0.53, 0.67, 1.00, 0.67,
    #                                    0.32, 0.36, 0.35, 0.38, 0.36, 0.39, 0.44, 0.53, 0.67, 1.00 }
    CorrelatedNoiseMatrix = cms.vdouble(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    doReadout = cms.bool(True),
    simHitToPhotoelectronsBarrel = cms.double(2250.0),
    syncPhase = cms.bool(False),
    doPhotostatistics = cms.bool(True),
    photoelectronsToAnalogBarrel = cms.double(0.000444444)
)


