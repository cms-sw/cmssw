import FWCore.ParameterSet.Config as cms

# configuration to model pileup for initial physics phase
from SimGeneral.MixingModule.aliases_cfi import *
from SimGeneral.MixingModule.pixelDigitizer_cfi import *
from SimGeneral.MixingModule.stripDigitizer_cfi import *
from SimGeneral.MixingModule.ecalDigitizer_cfi import *
from SimGeneral.MixingModule.hcalDigitizer_cfi import *
from SimGeneral.MixingModule.castorDigitizer_cfi import *
from SimGeneral.MixingModule.trackingTruthProducer_cfi import *
from SimGeneral.MixingModule.caloTruthProducer_cfi import *

pixelDigitizer.TofLowerCut=cms.double(18.5)
pixelDigitizer.TofUpperCut=cms.double(43.5)
stripDigitizer.CosmicDelayShift = cms.untracked.double(31)

ecalDigitizer.cosmicsPhase = cms.bool(True)
ecalDigitizer.cosmicsShift = cms.double(1.)

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

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify( theDigitizers, castor = None )

from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import hgceeDigitizer, hgchebackDigitizer, hgchefrontDigitizer, HGCAL_noise_fC, HGCAL_noise_heback, HFNose_noise_fC, HGCAL_chargeCollectionEfficiencies, HGCAL_ileakParam_toUse, HGCAL_cceParams_toUse, HGCAL_noises

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify( theDigitizers,
                       hgceeDigitizer = cms.PSet(hgceeDigitizer),
                       hgchebackDigitizer = cms.PSet(hgchebackDigitizer),
                       hgchefrontDigitizer = cms.PSet(hgchefrontDigitizer),
                       calotruth = cms.PSet(caloParticles), #HGCAL still needs calotruth for production mode
)

from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import hfnoseDigitizer

from Configuration.Eras.Modifier_phase2_hfnose_cff import phase2_hfnose
phase2_hfnose.toModify( theDigitizers,
                        hfnoseDigitizer = cms.PSet(hfnoseDigitizer),
)

from SimGeneral.MixingModule.ecalTimeDigitizer_cfi import ecalTimeDigitizer
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify( theDigitizers,
                        ecalTime = ecalTimeDigitizer.clone() )

from SimFastTiming.Configuration.SimFastTiming_cff import mtdDigitizer
from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
phase2_timing_layer.toModify( theDigitizers,
                              fastTimingLayer = mtdDigitizer.clone() )

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(theDigitizers,
                        strip = None)

theDigitizersValid = cms.PSet(theDigitizers,
                              mergedtruth = cms.PSet(trackingParticles))
