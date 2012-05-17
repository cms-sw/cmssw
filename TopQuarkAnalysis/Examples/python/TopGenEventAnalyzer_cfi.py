import FWCore.ParameterSet.Config as cms

#
# module to make simple analyses of muons
#
analyzeTopGenEvent = cms.EDAnalyzer("TopGenEventAnalyzer",
    genEvent = cms.InputTag("genEvt")
)


