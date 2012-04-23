import FWCore.ParameterSet.Config as cms

#
# module to make the kinematic fit hypothesis
#
ttSemiLepHypKinFit = cms.EDProducer("TtSemiLepHypKinFit",
    ## met input
    mets  = cms.InputTag("patMETs"),
    ## jet input
    jets  = cms.InputTag("selectedPatJets"),
    ## lepton input                    
    leps  = cms.InputTag("selectedPatMuons"),
    ## kin fit results
    match       = cms.InputTag("kinFitTtSemiLepEventHypothesis"),
    status      = cms.InputTag("kinFitTtSemiLepEventHypothesis","Status"),
    leptons     = cms.InputTag("kinFitTtSemiLepEventHypothesis","Leptons"),
    neutrinos   = cms.InputTag("kinFitTtSemiLepEventHypothesis","Neutrinos"),                                    
    partonsHadP = cms.InputTag("kinFitTtSemiLepEventHypothesis","PartonsHadP"),
    partonsHadQ = cms.InputTag("kinFitTtSemiLepEventHypothesis","PartonsHadQ"),
    partonsHadB = cms.InputTag("kinFitTtSemiLepEventHypothesis","PartonsHadB"),
    partonsLepB = cms.InputTag("kinFitTtSemiLepEventHypothesis","PartonsLepB")
)


