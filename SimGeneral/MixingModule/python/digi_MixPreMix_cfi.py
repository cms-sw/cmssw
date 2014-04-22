import FWCore.ParameterSet.Config as cms

# configuration to model pileup for initial physics phase
from SimGeneral.MixingModule.aliases_PreMix_cfi import *
from SimGeneral.MixingModule.pixelDigitizer_cfi import *
from SimGeneral.MixingModule.stripDigitizer_cfi import *
#from SimGeneral.MixingModule.ecalDigitizer_cfi import *
#from SimGeneral.MixingModule.hcalDigitizer_cfi import *
from SimGeneral.MixingModule.castorDigitizer_cfi import *
from SimGeneral.MixingModule.trackingTruthProducer_cfi import *

theDigitizersMixPreMix = cms.PSet(
  pixel = cms.PSet(
    pixelDigitizer
  ),
  strip = cms.PSet(
    stripDigitizer
  ),
  castor  = cms.PSet(
    castorDigitizer
  )
)


#theDigitizersNoNoise.pixel.AddNoise = cms.bool(True)
#theDigitizersNoNoise.pixel.addNoisyPixels = cms.bool(False)
theDigitizersMixPreMix.strip.Noise = cms.bool(False) # will be added in DataMixer

