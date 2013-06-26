import FWCore.ParameterSet.Config as cms

#
# module to filter events based on member functions of the TtSemiLeptonicEvent
#
ttSemiLepEventFilter = cms.EDFilter("TtSemiLepEvtFilter",
    src = cms.InputTag("ttSemiLepEvent"),
    cut = cms.string("isHypoValid('kGenMatch') & genMatchSumDR < 999.")
)
