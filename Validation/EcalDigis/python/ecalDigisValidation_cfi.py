import FWCore.ParameterSet.Config as cms

ecalDigisValidation = cms.EDFilter("EcalDigisValidation",
    outputFile = cms.untracked.string('EcalDigisValidation.root'),
    verbose = cms.untracked.bool(True),
    EBdigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    ESdigiCollection = cms.InputTag("ecalPreshowerDigis"),
    moduleLabelMC = cms.string('source'),
    EEdigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    moduleLabelG4 = cms.string('g4SimHits')
)


