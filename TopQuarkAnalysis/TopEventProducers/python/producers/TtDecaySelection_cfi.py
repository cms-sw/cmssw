import FWCore.ParameterSet.Config as cms

#
# module to perform a selection of specific top
# deecays based on the genEvt
#
ttDecaySelection = cms.EDFilter("TtDecaySelection",
    src = cms.InputTag("genEvt"),
    invert    = cms.bool(False),

    ## syntax is (electron, muon, tau), 1 allows
    ## the corresponding lepton 0 does not.                                
    channel_1 = cms.vint32(0, 0, 0), 
    channel_2 = cms.vint32(0, 0, 0),
    tauDecays = cms.vint32(0, 0, 0)
)


