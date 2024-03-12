import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalBarrelRecHitsValidation = DQMEDAnalyzer('EcalBarrelRecHitsValidation',
    EBdigiCollection = cms.InputTag("simEcalDigis","ebDigis"),
    EBuncalibrechitCollection = cms.InputTag("ecalMultiFitUncalibRecHit","EcalUncalibRecHitsEB"),
    verbose = cms.untracked.bool(False)
)



# foo bar baz
# OCSK3ftzem7A2
# tR5MhYQapcNYh
