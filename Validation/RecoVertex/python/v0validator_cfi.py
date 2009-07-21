import FWCore.ParameterSet.Config as cms

v0Validator = cms.EDAnalyzer('V0Validator',
    DQMRootFileName = cms.string('validation.sample.root'),
    dirName = cms.string('V0V')
)
