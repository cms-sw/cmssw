import FWCore.ParameterSet.Config as cms

MyMomConstraint = cms.EDProducer("MomentumConstraintProducer",
                                              src                = cms.InputTag("ALCARECOTkAlCosmicsCTF0T"),
                                              fixedMomentum      = cms.double(1.0),
                                              fixedMomentumError = cms.double(0.005)
                                              )
