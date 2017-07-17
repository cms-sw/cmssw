import FWCore.ParameterSet.Config as cms

# configuration to model pileup for initial physics phase
from SimGeneral.MixingModule.aliases_PreMix_cfi import *
from SimGeneral.MixingModule.pixelDigitizer_cfi import *
from SimGeneral.MixingModule.stripDigitizer_cfi import *
from SimGeneral.MixingModule.pileupVtxDigitizer_cfi import *
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
  ),
  puVtx = cms.PSet(
    pileupVtxDigitizer
  )
)

theDigitizersMixPreMix.strip.Noise = cms.bool(False) # will be added in DataMixer
theDigitizersMixPreMix.strip.PreMixingMode = cms.bool(True) #Special mode to save all hit strips
theDigitizersMixPreMix.strip.FedAlgorithm = cms.int32(5) # special ZS mode: accept adc>0
theDigitizersMixPreMix.pixel.AddPixelInefficiency = cms.bool(False) # will be added in DataMixer    

from Configuration.Eras.Modifier_fastSim_cff import fastSim
if fastSim.isChosen():
    # fastsim does not model castor
    delattr(theDigitizersMixPreMix,"castor")
    # fastsim does not digitize pixel and strip hits
    delattr(theDigitizersMixPreMix,"pixel")
    delattr(theDigitizersMixPreMix,"strip")

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify( theDigitizersMixPreMix, castor = None )
    
theDigitizersMixPreMixValid = cms.PSet(
    theDigitizersMixPreMix,
    mergedtruth = cms.PSet(
        trackingParticles
        )
    )

