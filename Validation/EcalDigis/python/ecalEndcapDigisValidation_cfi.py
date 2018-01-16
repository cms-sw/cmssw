import FWCore.ParameterSet.Config as cms

ecalEndcapDigisValidation = DQMStep1Module('EcalEndcapDigisValidation',
    EEdigiCollection = cms.InputTag("simEcalDigis","eeDigis"),
    verbose = cms.untracked.bool(False)
)


