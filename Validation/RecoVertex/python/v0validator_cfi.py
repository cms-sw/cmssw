import FWCore.ParameterSet.Config as cms

v0Validator = cms.EDAnalyzer('V0Validator',
    DQMRootFileName = cms.string('V0DQM.root')
)
