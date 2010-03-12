import FWCore.ParameterSet.Config as cms

#
# module to make the kinematic solution hypothesis
#
ttFullLepHypKinSolution = cms.EDProducer("TtFullLepHypKinSolution",
    electrons = cms.InputTag("selectedPatElectrons"),
    muons     = cms.InputTag("selectedPatMuons"),
    jets      = cms.InputTag("selectedPatJets"),    
    mets      = cms.InputTag("patMETs"),
    
    match          = cms.InputTag("kinSolutionTtFullLepEventHypothesis"),       
    Neutrinos      = cms.InputTag("kinSolutionTtFullLepEventHypothesis","fullLepNeutrinos"),       
    NeutrinoBars   = cms.InputTag("kinSolutionTtFullLepEventHypothesis","fullLepNeutrinoBars"), 
    solutionWeight = cms.InputTag("kinSolutionTtFullLepEventHypothesis","solWeight"),
    jetCorrectionLevel = cms.string("abs") 
)


