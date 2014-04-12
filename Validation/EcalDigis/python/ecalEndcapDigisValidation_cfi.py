import FWCore.ParameterSet.Config as cms

ecalEndcapDigisValidation = cms.EDAnalyzer("EcalEndcapDigisValidation",
    EEdigiCollection = cms.InputTag("simEcalDigis","eeDigis"),
    verbose = cms.untracked.bool(False)
)


