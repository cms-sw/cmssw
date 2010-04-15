import FWCore.ParameterSet.Config as cms

simEcalPreshowerDigis = cms.EDProducer("ESZeroSuppressionProducer",
    ESNoiseSigma = cms.untracked.double(6.0),
    ESMIPkeV = cms.untracked.double(81.08),
    ESGain = cms.untracked.int32(2),
    ESMIPADC = cms.untracked.double(55.0),
    ESdigiCollection = cms.string(''),
    ESBaseline = cms.untracked.int32(1000),
    digiProducer = cms.string('simEcalUnsuppressedDigis'),
    ESZSdigiCollection = cms.string('')
)



