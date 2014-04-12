import FWCore.ParameterSet.Config as cms

castorDigitizer = cms.PSet(
    accumulatorType = cms.string("CastorDigiProducer"),
    hitsProducer = cms.string("g4SimHits"),
    makeDigiSimLinks = cms.untracked.bool(False),
    doNoise = cms.bool(True),
    doTimeSlew = cms.bool(True),
    castor = cms.PSet(
        readoutFrameSize = cms.int32(6),
        binOfMaximum = cms.int32(5),
        samplingFactor = cms.double(0.060135), ## GeV/pe

        doPhotoStatistics = cms.bool(True),
        photoelectronsToAnalog = cms.double(4.009),
        simHitToPhotoelectrons = cms.double(1000.0),
        syncPhase = cms.bool(True),
        timePhase = cms.double(-4.0)
    )
)
