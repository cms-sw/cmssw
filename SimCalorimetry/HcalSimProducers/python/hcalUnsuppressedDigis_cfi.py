import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HcalSimProducers.hcalSimParameters_cfi import *
simHcalUnsuppressedDigis = cms.EDProducer("HcalDigiProducer",
    hcalSimParameters,
    doNoise = cms.bool(True),
    doHPDNoise = cms.bool(False),
    doTimeSlew = cms.bool(True),
    hitsProducer = cms.string('g4SimHits')
)



