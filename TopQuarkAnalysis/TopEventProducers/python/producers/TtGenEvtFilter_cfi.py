import FWCore.ParameterSet.Config as cms

#
# module to filter events based on member functions of the TtGenEvent
#
ttGenEventFilter = cms.EDFilter("TtGenEvtFilter",
    src = cms.InputTag("genEvt"),
    cut = cms.string("")
)
