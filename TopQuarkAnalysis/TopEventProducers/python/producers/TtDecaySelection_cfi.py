import FWCore.ParameterSet.Config as cms

ttDecaySelection = cms.EDFilter("TtDecaySelection",
    src = cms.InputTag("genEvt"),
    invert = cms.bool(False),
    channel_1 = cms.vint32(0, 0, 0),
    tauDecays = cms.vint32(0, 0, 0),
    channel_2 = cms.vint32(0, 0, 0)
)


