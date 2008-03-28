import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HcalSimProducers.hcalSimParameters_cfi import *
hcalUnsuppressedDigis = cms.EDProducer("HcalDigiProducer",
    hcalSimParameters,
    doNoise = cms.bool(True),
    doTimeSlew = cms.bool(True)
)


