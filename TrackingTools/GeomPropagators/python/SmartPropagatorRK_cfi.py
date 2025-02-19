import FWCore.ParameterSet.Config as cms

SmartPropagatorRK = cms.ESProducer("SmartPropagatorESProducer",
                                   ComponentName = cms.string('SmartPropagatorRK'),
                                   TrackerPropagator = cms.string('RungeKuttaTrackerPropagator'),
                                   MuonPropagator = cms.string('SteppingHelixPropagatorAlong'),
                                   PropagationDirection = cms.string('alongMomentum'),
                                   Epsilon = cms.double(5.0)
                                   )


