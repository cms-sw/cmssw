import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HcalSimProducers.hcalSimParameters_cfi import *
simHcalUnsuppressedDigis = cms.EDProducer("HcalDigiProducer",
    hcalSimParameters,
    doNoise = cms.bool(True),
    doHPDNoise = cms.bool(False),
    HPDNoiseLibrary = cms.PSet(
       FileName = cms.FileInPath("SimCalorimetry/HcalSimAlgos/data/hpdNoiseLibrary.root"),
       HPDName = cms.untracked.string("HPD")
    ),
    doTimeSlew = cms.bool(True),
    hitsProducer = cms.string('g4SimHits')
)



