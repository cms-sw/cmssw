import FWCore.ParameterSet.Config as cms

#
# module to filter events based on member functions of the TtFullLeptonicEvent
#
ttFullLepEventFilter = cms.EDFilter("TtFullLepEvtFilter",
    src = cms.InputTag("ttFullLepEvent"),
    cut = cms.string("isHypoValid('kGenMatch') & genMatchSumDR < 999.")
)
