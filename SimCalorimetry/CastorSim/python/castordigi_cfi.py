import FWCore.ParameterSet.Config as cms

simCastorDigis = cms.EDProducer("CastorDigiProducer",
    doNoise = cms.bool(True),
    doTimeSlew = cms.bool(True),
    castor = cms.PSet(
        readoutFrameSize = cms.int32(6),
        binOfMaximum = cms.int32(4),
        samplingFactor = cms.double(0.267), ## pe/GeV

        doPhotoStatistics = cms.bool(True),
        photoelectronsToAnalog = cms.double(1.843),
        simHitToPhotoelectrons = cms.double(6.0),
        syncPhase = cms.bool(True),
        timePhase = cms.double(-4.0)
    )
)


