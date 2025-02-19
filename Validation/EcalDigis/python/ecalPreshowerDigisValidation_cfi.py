import FWCore.ParameterSet.Config as cms

ecalPreshowerDigisValidation = cms.EDAnalyzer("EcalPreshowerDigisValidation",
    ESdigiCollection = cms.InputTag("simEcalPreshowerDigis"),
    verbose = cms.untracked.bool(False)
)


