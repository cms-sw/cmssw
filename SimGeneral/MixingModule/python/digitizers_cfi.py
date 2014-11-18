import FWCore.ParameterSet.Config as cms

# configuration to model pileup for initial physics phase
from SimGeneral.MixingModule.aliases_cfi import *
from SimGeneral.MixingModule.pixelDigitizer_cfi import *
from SimGeneral.MixingModule.stripDigitizer_cfi import *
from SimGeneral.MixingModule.ecalDigitizer_cfi import *
from SimGeneral.MixingModule.hcalDigitizer_cfi import *
from SimGeneral.MixingModule.castorDigitizer_cfi import *
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
  castor  = cms.PSet(
    castorDigitizer
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
  mergedtruth = cms.PSet(
    trackingParticles
  )
)

def modifyMixForPostLS1( mixInstance, digitizers=None ):
    """
    Modifies the MixingModule for running in the Run 2 scenario.
    Currently just changes the dynamic inefficiency in the pixel
    digitiser by calling a function defined where the pixel
    digitiser is.
    
    First parameter is the mixing module object.
    
    Second optional parameter is the digitizers object. If not
    supplied it is taken from "mixInstance.digitizers" (required
    so that theDigitizersValid can also be modified).
    """
    if digitizers==None : # Check to see if second parameter was provided, if not use default
        digitizers=mixInstance.digitizers
    if hasattr( digitizers, 'pixel' ):
        import SimGeneral.MixingModule.pixelDigitizer_cfi
        if mixInstance.bunchspace == 25 :
            SimGeneral.MixingModule.pixelDigitizer_cfi.modifyPixelDigitizerForRun2Bunchspacing25( digitizers.pixel, mixInstance.bunchspace )
        elif mixInstance.bunchspace == 50 :
            SimGeneral.MixingModule.pixelDigitizer_cfi.modifyPixelDigitizerForRun2Bunchspacing50( digitizers.pixel, mixInstance.bunchspace )



