import FWCore.ParameterSet.Config as cms

ecalBarrelDigisValidation = DQMStep1Module('EcalBarrelDigisValidation',
    EBdigiCollection = cms.InputTag("simEcalDigis","ebDigis"),
    verbose = cms.untracked.bool(False)
)


