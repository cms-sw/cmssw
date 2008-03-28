import FWCore.ParameterSet.Config as cms

ttSemiLeptonicFilter = cms.EDFilter("TtDecayChannelFilter",
    src = cms.InputTag("genParticles"),
    # invert selection?
    invert = cms.bool(False),
    #---------------------------------------
    # allowed families for first (channel_1)
    # and/or second (channel_2) lepton. In 
    # order: (elec,muon,tau)
    #---------------------------------------
    channel_1 = cms.vint32(1, 1, 0),
    channel_2 = cms.vint32(0, 0, 0)
)


