import FWCore.ParameterSet.Config as cms

simEcalGlobalZeroSuppression = cms.EDProducer("EcalZeroSuppressionProducer",
                                              unsuppressedEBDigiCollection = cms.InputTag("simEcalUnsuppressedDigis"),
                                              unsuppressedEEDigiCollection = cms.InputTag("simEcalUnsuppressedDigis"),
                                              unsuppressedEKDigiCollection = cms.InputTag("simEcalUnsuppressedDigis"),
                                              EBZSdigiCollection = cms.string(''),
                                              glbBarrelThreshold = cms.untracked.double(0),
                                              EEZSdigiCollection = cms.string(''),
                                              glbEndcapThreshold = cms.untracked.double(0),
                                              EKZSdigiCollection = cms.string('ekDigis'),
                                              glbShashlikThreshold = cms.untracked.double(1), #ADC counts of sampleMax > pedestal
)
