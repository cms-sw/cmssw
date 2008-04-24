import FWCore.ParameterSet.Config as cms

ecalDigisValidation = cms.EDFilter("EcalDigisValidation",
    outputFile = cms.untracked.string('EcalDigisValidation.root'),
    verbose = cms.untracked.bool(True),
    EBdigiCollection = cms.InputTag("simEcalDigis","ebDigis"),
    ESdigiCollection = cms.InputTag("simEcalPreshowerDigis"),
    moduleLabelMC = cms.string('source'),
    EEdigiCollection = cms.InputTag("simEcalDigis","eeDigis"),
    moduleLabelG4 = cms.string('g4SimHits')
)


