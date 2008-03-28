import FWCore.ParameterSet.Config as cms

ecalEndcapDigisValidation = cms.EDFilter("EcalEndcapDigisValidation",
    EEdigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    verbose = cms.untracked.bool(True)
)


