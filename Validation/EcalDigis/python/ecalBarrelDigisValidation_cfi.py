import FWCore.ParameterSet.Config as cms

ecalBarrelDigisValidation = cms.EDAnalyzer("EcalBarrelDigisValidation",
    EBdigiCollection = cms.InputTag("simEcalDigis","ebDigis"),
    verbose = cms.untracked.bool(False)
)


