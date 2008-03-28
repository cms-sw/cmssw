import FWCore.ParameterSet.Config as cms

metAnalyzer = cms.EDFilter("METTester",
    InputGenMETLabel = cms.string('genMet'),
    OutputFile = cms.untracked.string('METTester_data.root'),
    InputCaloMETLabel = cms.string('met')
)

recoMETValidation = cms.Sequence(metAnalyzer)

