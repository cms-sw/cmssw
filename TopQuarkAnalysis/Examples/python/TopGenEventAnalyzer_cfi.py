import FWCore.ParameterSet.Config as cms

#
# module to make simple analyses based on the TtGenEvent
#
analyzeTopGenEvent = cms.EDAnalyzer("TopGenEventAnalyzer",
    genEvent = cms.InputTag("genEvt")
)


