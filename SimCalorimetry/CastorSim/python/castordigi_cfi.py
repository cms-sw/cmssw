import FWCore.ParameterSet.Config as cms

simCastorDigis = cms.EDProducer("CastorDigiProducer",
    doNoise = cms.bool(True),
    doTimeSlew = cms.bool(True),
    castor = cms.PSet(
        readoutFrameSize = cms.int32(6),
        binOfMaximum = cms.int32(5),
        samplingFactor = cms.double(16.75), ## pe/GeV

        doPhotoStatistics = cms.bool(True),
        photoelectronsToAnalog = cms.double(0.212), #pPb: 0.848 = 4.24/5 => PbPb: 4.24/20 = 0.212
        simHitToPhotoelectrons = cms.double(1000.0),
        syncPhase = cms.bool(True),
        timePhase = cms.double(-4.0)
    )
)


