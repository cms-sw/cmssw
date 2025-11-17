import FWCore.ParameterSet.Config as cms

from Validation.EcalRecHits.ecalRecHitsValidation_cfi import *

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(ecalRecHitsValidation, hitsProducer = "fastSimProducer")

from Configuration.Eras.Modifier_phase2_ecal_devel_cff import phase2_ecal_devel
phase2_ecal_devel.toModify(ecalRecHitsValidation,
    EBuncalibrechitCollection = "ecalUncalibRecHitPhase2:EcalUncalibRecHitsEB",
    EEuncalibrechitCollection = None,
    EErechitCollection = None,
    ESrechitCollection = None,
    enableEndcaps = False
)
