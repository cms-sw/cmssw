import FWCore.ParameterSet.Config as cms

# configuration to model pileup for initial physics phase
from SimGeneral.MixingModule.aliases_cfi import *
from SimGeneral.MixingModule.pixelDigitizer_cfi import *
from SimGeneral.MixingModule.stripDigitizer_cfi import *
from SimGeneral.MixingModule.ecalDigitizer_cfi import *
from SimGeneral.MixingModule.hcalDigitizer_cfi import *
from SimGeneral.MixingModule.castorDigitizer_cfi import *
from SimGeneral.MixingModule.pileupVtxDigitizer_cfi import *
from SimGeneral.MixingModule.trackingTruthProducerSelection_cfi import *

theDigitizers = cms.PSet(
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
  castor = cms.PSet(
    castorDigitizer
  ),
  puVtx = cms.PSet(
    pileupVtxDigitizer
  )
)

theDigitizersValid = cms.PSet(
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





