import FWCore.ParameterSet.Config as cms

v0Validator = cms.EDAnalyzer('V0Validator',
    DQMRootFileName = cms.string(''),
    kShortCollection = cms.InputTag('generalV0Candidates:Kshort'),
    lambdaCollection = cms.InputTag('generalV0Candidates:Lambda'),
    dirName = cms.string('Vertexing/V0V')
)
