import FWCore.ParameterSet.Config as cms

#
# module to make the kinematic fit hypothesis
#
ttSemiLepHypKinFit = cms.EDProducer("TtSemiLepHypKinFit",
    ## met input
    mets  = cms.InputTag("layer1METs"),
    ## jet input
    jets  = cms.InputTag("selectedLayer1Jets"),
    ## lepton input                    
    leps  = cms.InputTag("selectedLayer1Muons"),
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


