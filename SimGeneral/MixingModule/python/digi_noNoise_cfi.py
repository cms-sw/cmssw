import FWCore.ParameterSet.Config as cms

# configuration to model pileup for initial physics phase
from SimGeneral.MixingModule.aliases_cfi import *
from SimGeneral.MixingModule.pixelDigitizer_cfi import *
from SimGeneral.MixingModule.stripDigitizer_cfi import *
from SimGeneral.MixingModule.ecalDigitizer_cfi import *
from SimGeneral.MixingModule.hcalDigitizer_cfi import *
from SimGeneral.MixingModule.castorDigitizer_cfi import *
from SimGeneral.MixingModule.trackingTruthProducer_cfi import *

theDigitizersNoNoise = cms.PSet(
  pixel = cms.PSet(
    pixelDigitizer
  ),
  strip = cms.PSet(
    stripDigitizer
  ),
  ecal = cms.PSet(
    ecalDigitizer
  ),
  hcal = cms.PSet(
    hcalDigitizer
  ),
  castor  = cms.PSet(
    castorDigitizer
  )
)


theDigitizersNoNoise.hcal.doNoise = cms.bool(False)
theDigitizersNoNoise.hcal.doEmpty = cms.bool(False)
theDigitizersNoNoise.hcal.doHPDNoise = cms.bool(False)
theDigitizersNoNoise.hcal.doIonFeedback = cms.bool(False)
theDigitizersNoNoise.hcal.doThermalNoise = cms.bool(False)
theDigitizersNoNoise.ecal.doNoise = cms.bool(False)
theDigitizersNoNoise.pixel.AddNoise = cms.bool(False)
theDigitizersNoNoise.strip.AddNoise = cms.bool(False)

