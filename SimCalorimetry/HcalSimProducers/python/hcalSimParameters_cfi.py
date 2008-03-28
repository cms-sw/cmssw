import FWCore.ParameterSet.Config as cms

hcalSimParameters = cms.PSet(
    hf1 = cms.PSet(
        readoutFrameSize = cms.int32(6),
        binOfMaximum = cms.int32(4),
        samplingFactor = cms.double(0.278),
        doPhotoStatistics = cms.bool(True),
        photoelectronsToAnalog = cms.double(2.79),
        simHitToPhotoelectrons = cms.double(6.0),
        syncPhase = cms.bool(True),
        timePhase = cms.double(4.0)
    ),
    hf2 = cms.PSet(
        readoutFrameSize = cms.int32(6),
        binOfMaximum = cms.int32(4),
        samplingFactor = cms.double(0.267),
        doPhotoStatistics = cms.bool(True),
        photoelectronsToAnalog = cms.double(1.843),
        simHitToPhotoelectrons = cms.double(6.0),
        syncPhase = cms.bool(True),
        timePhase = cms.double(4.0)
    ),
    ho = cms.PSet(
        readoutFrameSize = cms.int32(10),
        firstRing = cms.int32(1),
        binOfMaximum = cms.int32(5),
        doPhotoStatistics = cms.bool(True),
        photoelectronsToAnalog = cms.double(0.3065),
        simHitToPhotoelectrons = cms.double(4000.0),
        samplingFactors = cms.vdouble(217.0, 217.0, 217.0, 217.0, 217.0, 217.0, 217.0, 217.0, 217.0, 217.0, 217.0, 217.0, 217.0, 217.0, 217.0, 217.0),
        syncPhase = cms.bool(True),
        timePhase = cms.double(5.0)
    ),
    hb = cms.PSet(
        readoutFrameSize = cms.int32(10),
        firstRing = cms.int32(1),
        binOfMaximum = cms.int32(5),
        doPhotoStatistics = cms.bool(True),
        photoelectronsToAnalog = cms.double(0.3305),
        simHitToPhotoelectrons = cms.double(2000.0),
        samplingFactors = cms.vdouble(120.58, 120.58, 119.9, 120.14, 120.63, 120.21, 119.69, 119.55, 119.62, 119.6, 118.65, 119.38, 118.89, 120.34, 119.01, 132.37),
        syncPhase = cms.bool(True),
        timePhase = cms.double(5.0)
    ),
    zdc = cms.PSet(
        readoutFrameSize = cms.int32(6),
        binOfMaximum = cms.int32(4),
        samplingFactor = cms.double(0.267),
        doPhotoStatistics = cms.bool(True),
        photoelectronsToAnalog = cms.double(1.843),
        simHitToPhotoelectrons = cms.double(6.0),
        syncPhase = cms.bool(True),
        timePhase = cms.double(-4.0)
    ),
    he = cms.PSet(
        readoutFrameSize = cms.int32(10),
        firstRing = cms.int32(16),
        binOfMaximum = cms.int32(5),
        doPhotoStatistics = cms.bool(True),
        photoelectronsToAnalog = cms.double(0.3305),
        simHitToPhotoelectrons = cms.double(2000.0),
        samplingFactors = cms.vdouble(196.83, 188.3, 175.24, 177.2, 178.45, 177.39, 178.18, 177.6, 178.26, 179.63, 179.98, 180.52, 174.63, 174.63),
        syncPhase = cms.bool(True),
        timePhase = cms.double(5.0)
    )
)

