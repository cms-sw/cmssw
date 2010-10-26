import FWCore.ParameterSet.Config as cms

simHcalDigis = cms.EDProducer("HcalTBDigiProducer",
    doNoise             = cms.bool(False),
    doTimeSlew          = cms.bool(True),
    readoutFrameSizeTB  = cms.untracked.int32(10),
    doPhotostatisticsTB = cms.untracked.bool(True),
    syncPhaseTB         = cms.untracked.bool(False),
    tunePhaseShiftTB    = cms.untracked.double(1.0),
    photomultiplierGainTBHB    = cms.untracked.double(2000.0),
    photoelectronsToAnalogTBHB = cms.untracked.vdouble([0.3305]*16),
    samplingFactorTBHB         = cms.untracked.double(117.0),
    timePhaseTBHB              = cms.untracked.double(5.0),
    binOfMaximumTBHB           = cms.untracked.int32(5),
    firstRingTBHB              = cms.untracked.int32(1),
    samplingFactorsTBHB        = cms.vdouble(
            118.98, 118.60, 118.97, 118.76, 119.13,
            118.74, 117.80, 118.14, 116.87, 117.87,
            117.46, 116.79, 117.15, 117.29, 118.41,
            134.86),
    photomultiplierGainTBHE    = cms.untracked.double(2000.0),
    photoelectronsToAnalogTBHE = cms.vdouble([0.3305]*14),
    samplingFactorTBHE         = cms.untracked.double(178.0),
    timePhaseTBHE              = cms.untracked.double(5.0),
    binOfMaximumTBHE           = cms.untracked.int32(5),
    firstRingTBHE              = cms.untracked.int32(16),
    samplingFactorsTBHE        = cms.vdouble(
            197.84, 184.67, 170.60, 172.06, 173.08,
            171.92, 173.00, 173.22, 173.72, 174.21,
            173.91, 175.88, 171.65, 171.65),
    photomultiplierGainTBHO    = cms.untracked.double(4000.0),
    photoelectronsToAnalogTBHO = cms.vdouble(0.24, 0.24, 0.24, 0.24,
            0.17, 0.17, 0.17, 0.17, 0.17, 0.17,
            0.17, 0.17, 0.17, 0.17, 0.17),
    samplingFactorTBHO         = cms.untracked.double(217.0),
    timePhaseTBHO              = cms.untracked.double(5.0),
    binOfMaximumTBHO           = cms.untracked.int32(5),
    firstRingTBHO              = cms.untracked.int32(16),
    samplingFactorsTBHO        = cms.vdouble(231.0, 231.0, 231.0, 231.0,
            360.0, 360.0, 360.0, 360.0, 360.0, 360.0,
            360.0, 360.0, 360.0, 360.0, 360.0)
)



