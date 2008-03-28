import FWCore.ParameterSet.Config as cms

ecalBarrelDigisValidation = cms.EDFilter("EcalBarrelDigisValidation",
    EBdigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    verbose = cms.untracked.bool(True)
)


