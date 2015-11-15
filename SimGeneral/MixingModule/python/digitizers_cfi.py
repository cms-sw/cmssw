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
from FastSimulation.Tracking.recoTrackAccumulator_cfi import *

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

from Configuration.StandardSequences.Eras import eras
if eras.fastSim.isChosen():
    # fastsim does not model castor
    delattr(theDigitizers,"castor")
    # fastsim does not digitize pixel and strip hits
    delattr(theDigitizers,"pixel")
    delattr(theDigitizers,"strip")
    setattr(theDigitizers,"tracks",recoTrackAccumulator)
    
theDigitizersValid = cms.PSet(
    theDigitizers,
    mergedtruth = cms.PSet(
        trackingParticles
        )
    )
