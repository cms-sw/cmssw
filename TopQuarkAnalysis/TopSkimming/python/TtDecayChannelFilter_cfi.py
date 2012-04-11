import FWCore.ParameterSet.Config as cms

ttDecayChannelFilter = cms.EDFilter("TtDecayChannelFilter",
    ## input source for decay channel selection
    src = cms.InputTag("genParticles"),
    ## invert the selection choice                                
    invert = cms.bool(False),

    ## allow given lepton in corresponding decay
    ## branch for a given decay-channel selection;
    ## all leptons to 'False' corresponds to the
    ## fully-hadronic decay channel
    allowedTopDecays = cms.PSet(
      decayBranchA = cms.PSet(
        electron = cms.bool(False),
        muon     = cms.bool(False),
        tau      = cms.bool(False)
      ),
      decayBranchB= cms.PSet(
        electron = cms.bool(False),
        muon     = cms.bool(False),
        tau      = cms.bool(False)
      )
    ),

    ## add a restriction to the decay channel of taus
    ## by redefining the following ParameterSet in
    ## your cfg file; the following restrictions are
    ## available:
    restrictTauDecays = cms.PSet(
    #  electron   = cms.bool(True),
    #  muon       = cms.bool(True),
    #  oneProng   = cms.bool(True),
    #  threeProng = cms.bool(True)
    )
)


