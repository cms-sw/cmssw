import FWCore.ParameterSet.Config as cms

simpleMultiRecHitCollector = cms.ESProducer("MultiRecHitCollectorESProducer",
    propagatorAlong = cms.string('RungeKuttaTrackerPropagator'),
    MultiRecHitUpdator = cms.string('SiTrackerMultiRecHitUpdator'),
    ComponentName = cms.string('simpleMultiRecHitCollector'),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('RelaxedChi2Simple'),
    Mode = cms.string('Simple'),
    Debug = cms.bool(False)	
)


# foo bar baz
# 9VqB5IbsuqB1B
# bew1rHplCPNeC
