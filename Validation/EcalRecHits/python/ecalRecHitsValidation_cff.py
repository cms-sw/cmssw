import FWCore.ParameterSet.Config as cms

from Validation.EcalRecHits.ecalRecHitsValidation_cfi import *

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(ecalRecHitsValidation, hitsProducer = "fastSimProducer")

ecalRecHitsValidationPhase2 = ecalRecHitsValidation.clone(
    EBuncalibrechitCollection = "ecalUncalibRecHitPhase2:EcalUncalibRecHitsEB",
    EEuncalibrechitCollection = None,
    EErechitCollection = None,
    ESrechitCollection = None,
    enableEndcaps = False
)
