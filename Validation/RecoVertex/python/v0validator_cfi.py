import FWCore.ParameterSet.Config as cms

v0Validator = cms.EDAnalyzer('V0Validator',
    DQMRootFileName = cms.string('validation.sample.root'),
    kShortCollection = cms.InputTag('generalV0Candidates:Kshort'),
    lambdaCollection = cms.InputTag('generalV0Candidates:Lambda'),
    dirName = cms.string('V0V')
)
