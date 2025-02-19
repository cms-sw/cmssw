import FWCore.ParameterSet.Config as cms

SmartPropagatorAnyRKOpposite = cms.ESProducer("SmartPropagatorESProducer",
                                              ComponentName = cms.string('SmartPropagatorAnyRKOpposite'),
                                              TrackerPropagator = cms.string('RungeKuttaTrackerPropagatorOpposite'),
                                              MuonPropagator = cms.string('SteppingHelixPropagatorAny'),
                                              PropagationDirection = cms.string('oppositeToMomentum'),
                                              Epsilon = cms.double(5.0)
                                              )


