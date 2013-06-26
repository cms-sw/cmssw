import FWCore.ParameterSet.Config as cms

simEcalPreshowerDigis = cms.EDProducer("ESZeroSuppressionProducer",
    ESdigiCollection = cms.string(''),
    digiProducer = cms.string('simEcalUnsuppressedDigis'),
    ESZSdigiCollection = cms.string('')
)



