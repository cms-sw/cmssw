import FWCore.ParameterSet.Config as cms

ecalShashlikDigisValidation = cms.EDAnalyzer("EcalShashlikDigisValidation",
                                             #EKdigiCollection = cms.InputTag("simEcalDigis","ekDigis"),
                                             #EKdigiCollection = cms.InputTag("simEcalUnsuppressedDigis"),
                                             EKdigiCollection = cms.InputTag("simEcalGlobalZeroSuppression"),
    verbose = cms.untracked.bool(False)
)


