import FWCore.ParameterSet.Config as cms

cscNeutronWriter = cms.EDProducer("CSCNeutronWriter",
    timeWindow = cms.double(20.0),
    nChamberTypes = cms.int32(10),
    writer = cms.string('EDM'),
    output = cms.string('CSCNeutronHits.root'),
    neutronTimeCut = cms.double(250.0),
    # save the hits starting at 19 ns
    t0 = cms.double(19.),
    input = cms.InputTag("g4SimHits","MuonCSCHits")
)



