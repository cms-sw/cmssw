import FWCore.ParameterSet.Config as cms

L1Validator = cms.EDAnalyzer('L1Validator',
#  dirName=cms.string("L1T/L1T"),
  fileName=cms.string("L1Validation.root") #output file name
)
