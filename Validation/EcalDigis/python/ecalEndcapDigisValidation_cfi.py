import FWCore.ParameterSet.Config as cms

ecalEndcapDigisValidation = cms.EDFilter("EcalEndcapDigisValidation",
    EEdigiCollection = cms.InputTag("simEcalDigis","eeDigis"),
    verbose = cms.untracked.bool(True)
)


