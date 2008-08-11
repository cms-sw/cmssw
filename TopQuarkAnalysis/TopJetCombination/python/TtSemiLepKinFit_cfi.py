import FWCore.ParameterSet.Config as cms

#
# module to make the kinematic fit hypothesis
#
ttSemiLepKinFit = cms.EDProducer("TtSemiLepKinFit",
    jets  = cms.InputTag("selectedLayer1Jets"),
    leps  = cms.InputTag("selectedLayer1Muons"),
    mets  = cms.InputTag("selectedLayer1METs"),
    match = cms.InputTag("kinFitTtSemiEvent"),
    status    = cms.InputTag("kinFitTtSemiEvent","Status"),
    partons   = cms.InputTag("kinFitTtSemiEvent","Partons"),
    leptons   = cms.InputTag("kinFitTtSemiEvent","Leptons"),
    neutrinos = cms.InputTag("kinFitTtSemiEvent","Neutrinos")
)


