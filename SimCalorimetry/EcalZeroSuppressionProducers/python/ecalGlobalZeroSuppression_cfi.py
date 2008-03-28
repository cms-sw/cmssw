import FWCore.ParameterSet.Config as cms

ecalGlobalZeroSuppression = cms.EDProducer("EcalZeroSuppressionProducer",
    glbBarrelThreshold = cms.untracked.double(3.0),
    EEdigiCollection = cms.string(''),
    EBdigiCollection = cms.string(''),
    EEZSdigiCollection = cms.string(''),
    glbEndcapThreshold = cms.untracked.double(3.0),
    digiProducer = cms.string('ecalUnsuppressedDigis'),
    EBZSdigiCollection = cms.string('')
)


