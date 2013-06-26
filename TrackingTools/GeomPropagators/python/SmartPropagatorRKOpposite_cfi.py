import FWCore.ParameterSet.Config as cms

SmartPropagatorRKOpposite = cms.ESProducer("SmartPropagatorESProducer",
                                           ComponentName = cms.string('SmartPropagatorRKOpposite'),
                                           TrackerPropagator = cms.string('RungeKuttaTrackerPropagatorOpposite'),
                                           MuonPropagator = cms.string('SteppingHelixPropagatorOpposite'),
                                           PropagationDirection = cms.string('oppositeToMomentum'),
                                           Epsilon = cms.double(5.0)
                                           )


