import FWCore.ParameterSet.Config as cms

dtNeutronWriter = cms.EDProducer("DTNeutronWriter",
    timeWindow = cms.double(20.0),
    writer = cms.string('EDM'),
    output = cms.string('DTNeutronHits.root'),
    neutronTimeCut = cms.double(250.0),
    # save the hits starting at 13 ns
    t0 = cms.double(13.),
    input = cms.InputTag("g4SimHits","MuonDTHits")
)



