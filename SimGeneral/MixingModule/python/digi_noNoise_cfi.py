import FWCore.ParameterSet.Config as cms

# configuration to model pileup for initial physics phase
from SimGeneral.MixingModule.aliases_cfi import *
from SimGeneral.MixingModule.pixelDigitizer_cfi import *
from SimGeneral.MixingModule.stripDigitizer_cfi import *
from SimGeneral.MixingModule.ecalDigitizer_cfi import *
from SimGeneral.MixingModule.hcalDigitizer_cfi import *
from SimGeneral.MixingModule.castorDigitizer_cfi import *
from SimGeneral.MixingModule.pileupVtxDigitizer_cfi import *
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
  ),
  puVtx = cms.PSet(
    pileupVtxDigitizer
  )
)

theDigitizersNoNoise.hcal.doNoise = cms.bool(False)
theDigitizersNoNoise.hcal.doEmpty = cms.bool(False)
theDigitizersNoNoise.hcal.doHPDNoise = cms.bool(False)
theDigitizersNoNoise.hcal.doIonFeedback = cms.bool(False)
theDigitizersNoNoise.hcal.doThermalNoise = cms.bool(False)
theDigitizersNoNoise.hcal.doTimeSlew = cms.bool(False)
theDigitizersNoNoise.ecal.doENoise = cms.bool(False)
theDigitizersNoNoise.ecal.doESNoise = cms.bool(False)
theDigitizersNoNoise.ecal.applyConstantTerm = cms.bool(False)
theDigitizersNoNoise.pixel.AddNoise = cms.bool(True)
theDigitizersNoNoise.pixel.addNoisyPixels = cms.bool(False)
theDigitizersNoNoise.pixel.AddPixelInefficiency = cms.bool(False) #done in second step
theDigitizersNoNoise.strip.Noise = cms.bool(False)
theDigitizersNoNoise.strip.PreMixingMode = cms.bool(True)
theDigitizersNoNoise.strip.FedAlgorithm = cms.int32(5) # special ZS mode: accept adc>0
theDigitizersNoNoise.ecal.EcalPreMixStage1 = cms.bool(True)
theDigitizersNoNoise.hcal.HcalPreMixStage1 = cms.bool(True)

theDigitizersNoNoiseValid = cms.PSet(
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
  ),
  puVtx = cms.PSet(
    pileupVtxDigitizer
  ),
  mergedtruth = cms.PSet(
    trackingParticles
  )
)

theDigitizersNoNoiseValid.hcal.doNoise = cms.bool(False)
theDigitizersNoNoiseValid.hcal.doEmpty = cms.bool(False)
theDigitizersNoNoiseValid.hcal.doHPDNoise = cms.bool(False)
theDigitizersNoNoiseValid.hcal.doIonFeedback = cms.bool(False)
theDigitizersNoNoiseValid.hcal.doThermalNoise = cms.bool(False)
theDigitizersNoNoiseValid.hcal.doTimeSlew = cms.bool(False)
theDigitizersNoNoiseValid.ecal.doENoise = cms.bool(False)
theDigitizersNoNoiseValid.ecal.doESNoise = cms.bool(False)
theDigitizersNoNoiseValid.ecal.applyConstantTerm = cms.bool(False)
theDigitizersNoNoiseValid.pixel.AddNoise = cms.bool(True)
theDigitizersNoNoiseValid.pixel.addNoisyPixels = cms.bool(False)
theDigitizersNoNoiseValid.pixel.AddPixelInefficiency = cms.bool(False) #done in second step
theDigitizersNoNoiseValid.strip.Noise = cms.bool(False)
theDigitizersNoNoiseValid.strip.PreMixingMode = cms.bool(True)
theDigitizersNoNoiseValid.strip.FedAlgorithm = cms.int32(5) # special ZS mode: accept adc>0
theDigitizersNoNoiseValid.ecal.EcalPreMixStage1 = cms.bool(True)
theDigitizersNoNoiseValid.hcal.HcalPreMixStage1 = cms.bool(True)



