import FWCore.ParameterSet.Config as cms

genPUProtons = cms.EDProducer("GenPUProtonProducer",
	mix = cms.string("mix"),
	bunchCrossingList = cms.vint32(0),
	minPz = cms.double( 2400. )
	)
