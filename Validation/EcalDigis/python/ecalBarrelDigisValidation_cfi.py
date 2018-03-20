import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalBarrelDigisValidation = DQMEDAnalyzer('EcalBarrelDigisValidation',
    EBdigiCollection = cms.InputTag("simEcalDigis","ebDigis"),
    verbose = cms.untracked.bool(False)
)


