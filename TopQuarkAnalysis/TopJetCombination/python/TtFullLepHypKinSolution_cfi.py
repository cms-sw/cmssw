import FWCore.ParameterSet.Config as cms

#
# module to make the kinematic solution hypothesis
#
ttFullLepHypKinSolution = cms.EDProducer("TtFullLepHypKinSolution",
    electrons = cms.InputTag("selectedLayer1Electrons"),
    muons     = cms.InputTag("selectedLayer1Muons"),
    jets      = cms.InputTag("selectedLayer1Jets"),    
    mets      = cms.InputTag("layer1METs"),

    match          = cms.InputTag("kinSolutionTtFullLepEventHypothesis"),       
    Neutrinos      = cms.InputTag("kinSolutionTtFullLepEventHypothesis","fullLepNeutrinos"),       
    NeutrinoBars   = cms.InputTag("kinSolutionTtFullLepEventHypothesis","fullLepNeutrinoBars"), 
    solutionWeight = cms.InputTag("kinSolutionTtFullLepEventHypothesis","solWeight")
)


