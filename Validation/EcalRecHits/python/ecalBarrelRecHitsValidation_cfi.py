import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalBarrelRecHitsValidation = DQMEDAnalyzer('EcalBarrelRecHitsValidation',
    EBdigiCollection = cms.InputTag("simEcalDigis","ebDigis"),
    EBuncalibrechitCollection = cms.InputTag("ecalMultiFitUncalibRecHit","EcalUncalibRecHitsEB"),
    verbose = cms.untracked.bool(False)
)

from Configuration.Eras.Modifier_phase2_ecal_devel_cff import phase2_ecal_devel
phase2_ecal_devel.toModify(ecalBarrelRecHitsValidation,
     EBuncalibrechitCollection = "ecalUncalibRecHitPhase2:EcalUncalibRecHitsEB"
)
