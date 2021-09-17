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
  ),
  mergedtruth = cms.PSet(
    trackingParticles
  )
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(theDigitizers,
    # fastsim does not model castor
    castor = None,
    # fastsim does not digitize pixel and strip hits
    pixel = None,
    strip = None,
    tracks = recoTrackAccumulator
)
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
(fastSim & premix_stage2).toModify(theDigitizers,
    tracks = None
)


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

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify( theDigitizers, castor = None )

from SimGeneral.MixingModule.ecalTimeDigitizer_cfi import ecalTimeDigitizer
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify( theDigitizers,
                        ecalTime = ecalTimeDigitizer.clone() )

from SimFastTiming.Configuration.SimFastTiming_cff import mtdDigitizer
from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
phase2_timing_layer.toModify( theDigitizers,
                              fastTimingLayer = mtdDigitizer.clone() )

premix_stage2.toModify(theDigitizers,
    ecal = None,
    hcal = None,
)
(premix_stage2 & phase2_hgcal).toModify(theDigitizers,
    hgceeDigitizer = dict(premixStage1 = True),
    hgchebackDigitizer = dict(premixStage1 = True),
    hgchefrontDigitizer = dict(premixStage1 = True),
    calotruth = dict(premixStage1 = True), #HGCAL still needs calotruth for production mode
)
(premix_stage2 & phase2_hfnose).toModify(theDigitizers,
    hfnoseDigitizer = dict(premixStage1 = True),
)
(premix_stage2 & phase2_timing_layer).toModify(theDigitizers,
    fastTimingLayer = dict(
        barrelDigitizer = dict(premixStage1 = True),
        endcapDigitizer = dict(premixStage1 = True)
    )
)

theDigitizersValid = cms.PSet(theDigitizers)
theDigitizers.mergedtruth.select.signalOnlyTP = True

from Configuration.ProcessModifiers.run3_ecalclustering_cff import run3_ecalclustering
run3_ecalclustering.toModify( theDigitizersValid, 
                              calotruth = cms.PSet( caloParticles ) )

phase2_timing.toModify( theDigitizersValid.mergedtruth,
                        createInitialVertexCollection = cms.bool(True) )

from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
def _customizePremixStage1(mod):
    # To avoid this if-else structure we'd need an "_InverseModifier"
    # to customize pixel/strip for everything else than fastSim.
    if hasattr(mod, "pixel"):
        if hasattr(mod.pixel, "AlgorithmCommon"):
            mod.pixel.AlgorithmCommon.makeDigiSimLinks = True
        else:
            mod.pixel.makeDigiSimLinks = True
    if hasattr(mod, "strip"):
        mod.strip.makeDigiSimLinks = True
    mod.mergedtruth.select.signalOnlyTP = False
premix_stage1.toModify(theDigitizersValid, _customizePremixStage1)

def _loadPremixStage2Aliases(process):
    process.load("SimGeneral.MixingModule.aliases_PreMix_cfi")
modifyDigitizers_loadPremixStage2Aliases = premix_stage2.makeProcessModifier(_loadPremixStage2Aliases)
