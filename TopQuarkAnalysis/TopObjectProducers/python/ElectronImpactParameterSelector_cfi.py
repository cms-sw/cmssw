import FWCore.ParameterSet.Config as cms

impactParameterElectrons = cms.EDProducer("ElectronImpactParameterSelector",
    vertices = cms.InputTag("offlinePrimaryVertices"),
    leptons  = cms.InputTag("selectedPatElectrons"),
    cut      = cms.double(3)
)
