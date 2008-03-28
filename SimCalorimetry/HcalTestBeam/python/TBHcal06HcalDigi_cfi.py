import FWCore.ParameterSet.Config as cms

hcalDigis = cms.EDProducer("HcalTBDigiProducer",
    doPhotostatisticsTB = cms.untracked.bool(True),
    photomultiplierGainTBHO = cms.untracked.double(4000.0),
    tunePhaseShiftTB = cms.untracked.double(1.0),
    timePhaseTBHB = cms.untracked.double(5.0),
    samplingFactorTBHO = cms.untracked.double(217.0),
    photomultiplierGainTBHE = cms.untracked.double(2000.0),
    doNoise = cms.bool(False),
    samplingFactorTBHE = cms.untracked.double(178.0),
    samplingFactorTBHB = cms.untracked.double(117.0),
    photomultiplierGainTBHB = cms.untracked.double(2000.0),
    amplifierGainTBHB = cms.untracked.double(0.3305),
    timePhaseTBHO = cms.untracked.double(5.0),
    readoutFrameSizeTB = cms.untracked.int32(10),
    timePhaseTBHE = cms.untracked.double(5.0),
    doTimeSlew = cms.bool(True),
    amplifierGainTBHO = cms.untracked.double(0.3065),
    binOfMaximumTBHB = cms.untracked.int32(5),
    binOfMaximumTBHO = cms.untracked.int32(5),
    amplifierGainTBHE = cms.untracked.double(0.3305),
    syncPhaseTB = cms.untracked.bool(True),
    binOfMaximumTBHE = cms.untracked.int32(5)
)


