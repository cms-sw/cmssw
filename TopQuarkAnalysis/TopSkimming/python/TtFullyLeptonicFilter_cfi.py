import FWCore.ParameterSet.Config as cms

ttFullyLeptonicFilter = cms.EDFilter("TtDecayChannelFilter",
    src = cms.InputTag("genParticles"),
    #invert selection?
    invert = cms.bool(False),
    #---------------------------------------
    # select channels of consideration in
    # order: (elec,muon,tau)
    #---------------------------------------
    channel_1 = cms.vint32(1, 1, 0),
    #---------------------------------------
    # allowed decay channels for taus.
    # In order: (leptonic, one prong, three prong)
    #---------------------------------------
    tauDecays = cms.vint32(0, 1, 1),
    channel_2 = cms.vint32(1, 1, 0)
)


