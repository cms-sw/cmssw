import FWCore.ParameterSet.Config as cms

ecalPreshowerDigisValidation = cms.EDFilter("EcalPreshowerDigisValidation",
    ESdigiCollection = cms.InputTag("simEcalPreshowerDigis"),
    verbose = cms.untracked.bool(True)
)


