import FWCore.ParameterSet.Config as cms

g4SimHits = cms.EDProducer("NeutronHitsCollector",
    neutronLabelCSC = cms.untracked.string('cscNeutronWriter'),
    neutronLabelRPC = cms.untracked.string('rpcNeutronWriter'),
    neutronLabelDT = cms.untracked.string('dtNeutronWriter')
)
