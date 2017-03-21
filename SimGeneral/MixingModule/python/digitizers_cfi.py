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
from SimGeneral.MixingModule.caloTruthProducer_cfi import *
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

from Configuration.Eras.Modifier_fastSim_cff import fastSim
if fastSim.isChosen():
    # fastsim does not model castor
    delattr(theDigitizers,"castor")
    # fastsim does not digitize pixel and strip hits
    delattr(theDigitizers,"pixel")
    delattr(theDigitizers,"strip")
    setattr(theDigitizers,"tracks",recoTrackAccumulator)


from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import hgceeDigitizer, hgchebackDigitizer, hgchefrontDigitizer 
    
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify( theDigitizers,
                            hgceeDigitizer = cms.PSet(hgceeDigitizer),
                            hgchebackDigitizer = cms.PSet(hgchebackDigitizer),
                            hgchefrontDigitizer = cms.PSet(hgchefrontDigitizer),
)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify( theDigitizers, castor = None )

from SimGeneral.MixingModule.ecalTimeDigitizer_cfi import ecalTimeDigitizer
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
phase2_timing.toModify( theDigitizers,
                        ecalTime = ecalTimeDigitizer.clone() )
    
from SimFastTiming.Configuration.SimFastTiming_cff import fastTimeDigitizer
phase2_timing_layer.toModify( theDigitizers,
                        fastTimingLayer = fastTimeDigitizer.clone() )

theDigitizersValid = cms.PSet(
    theDigitizers,
    mergedtruth = cms.PSet(
        trackingParticles
        )
    )


phase2_hgcal.toModify( theDigitizersValid,
                       calotruth = cms.PSet( caloParticles ) )


phase2_timing.toModify( theDigitizersValid.mergedtruth,
                        createInitialVertexCollection = cms.bool(True) )

