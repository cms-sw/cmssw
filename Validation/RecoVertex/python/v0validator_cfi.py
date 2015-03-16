import FWCore.ParameterSet.Config as cms

v0Validator = cms.EDAnalyzer('V0Validator',
    DQMRootFileName = cms.untracked.string(''),
    kShortCollection = cms.untracked.InputTag('generalV0Candidates:Kshort'),
    lambdaCollection = cms.untracked.InputTag('generalV0Candidates:Lambda'),
    dirName = cms.untracked.string('Vertexing/V0V')
)
