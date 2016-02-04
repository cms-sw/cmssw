import FWCore.ParameterSet.Config as cms

#
# module to perform a selection of specific top
# deecays based on the genEvt
#
ttDecaySelection = cms.EDFilter("TtDecaySelection",
    ## input source for decay channel selection
    src    = cms.InputTag("genEvt"),
    ## invert the selection choice                                
    invert = cms.bool(False),

    ## allow given lepton in corresponding decay
    ## branch for a given decay channel selection;
    ## all leptons to 'False' corresponds to the
    ## full hadronic decay channel
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
    #  leptonic   = cms.bool(False),
    #  oneProng   = cms.bool(False),
    #  threeProng = cms.bool(False)
    )
)

