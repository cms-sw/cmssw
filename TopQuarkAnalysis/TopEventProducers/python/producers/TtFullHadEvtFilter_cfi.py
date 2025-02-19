import FWCore.ParameterSet.Config as cms

#
# module to filter events based on member functions of the TtFullHadronicEvent
#
ttFullHadEventFilter = cms.EDFilter("TtFullHadEvtFilter",
    src = cms.InputTag("ttFullHadEvent"),
    cut = cms.string("isHypoValid('kGenMatch') & genMatchSumDR < 999.")
)
