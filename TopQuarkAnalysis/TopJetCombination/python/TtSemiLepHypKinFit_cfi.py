import FWCore.ParameterSet.Config as cms

#
# module to make the kinematic fit hypothesis
#
ttSemiLepHypKinFit = cms.EDProducer("TtSemiLepHypKinFit",
    jets  = cms.InputTag("selectedLayer1Jets"),
    leps  = cms.InputTag("selectedLayer1Muons"),
    mets  = cms.InputTag("selectedLayer1METs"),
    match = cms.InputTag("kinFitTtSemiLepEvent"),
    status    = cms.InputTag("kinFitTtSemiLepEvent","Status"),
    partons   = cms.InputTag("kinFitTtSemiLepEvent","Partons"),
    leptons   = cms.InputTag("kinFitTtSemiLepEvent","Leptons"),
    neutrinos = cms.InputTag("kinFitTtSemiLepEvent","Neutrinos")
)


