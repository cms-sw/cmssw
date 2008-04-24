import FWCore.ParameterSet.Config as cms

ecalBarrelDigisValidation = cms.EDFilter("EcalBarrelDigisValidation",
    EBdigiCollection = cms.InputTag("simEcalDigis","ebDigis"),
    verbose = cms.untracked.bool(True)
)


