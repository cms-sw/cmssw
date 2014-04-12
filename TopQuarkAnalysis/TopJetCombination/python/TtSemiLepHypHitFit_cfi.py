import FWCore.ParameterSet.Config as cms

#
# module to make the kinematic fit hypothesis
#
ttSemiLepHypHitFit = cms.EDProducer("TtSemiLepHypHitFit",
    ## met input
    mets  = cms.InputTag("patMETs"),
    ## jet input
    jets  = cms.InputTag("selectedPatJetsAK5PF"),
    ## lepton input                    
    leps  = cms.InputTag("selectedPatMuons"),
    ## kin fit results
    match       = cms.InputTag("hitFitTtSemiLepEventHypothesis"),
    status      = cms.InputTag("hitFitTtSemiLepEventHypothesis","Status"),
    leptons     = cms.InputTag("hitFitTtSemiLepEventHypothesis","Leptons"),
    neutrinos   = cms.InputTag("hitFitTtSemiLepEventHypothesis","Neutrinos"),                                    
    partonsHadP = cms.InputTag("hitFitTtSemiLepEventHypothesis","PartonsHadP"),
    partonsHadQ = cms.InputTag("hitFitTtSemiLepEventHypothesis","PartonsHadQ"),
    partonsHadB = cms.InputTag("hitFitTtSemiLepEventHypothesis","PartonsHadB"),
    partonsLepB = cms.InputTag("hitFitTtSemiLepEventHypothesis","PartonsLepB"),
    ## number of considered jets
    nJetsConsidered = cms.InputTag("hitFitTtSemiLepEventHypothesis","NumberOfConsideredJets")
)


