import FWCore.ParameterSet.Config as cms

ESDigisReferenceDistrib = cms.EDAnalyzer("ESDigisReferenceDistrib",
    verbose = cms.untracked.bool(True),
    ESdigiCollection = cms.InputTag("simEcalPreshowerDigis"),
    outputTxtFile = cms.untracked.string('esRefHistosFile.txt'),
    outputRootFile = cms.untracked.string('esRefHistosFile.root')
)


