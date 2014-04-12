import FWCore.ParameterSet.Config as cms

rpcNeutronWriter = cms.EDProducer("RPCNeutronWriter",
    timeWindow = cms.double(20.0),
    writer = cms.string('EDM'),
    output = cms.string('RPCNeutronHits.root'),
    neutronTimeCut = cms.double(250.0),
    # save the hits starting at 19 ns
    t0 = cms.double(19.),
    input = cms.InputTag("g4SimHits","MuonRPCHits")
)



