import FWCore.ParameterSet.Config as cms

groupedMultiRecHitCollector = cms.ESProducer("MultiRecHitCollectorESProducer",
    propagatorAlong = cms.string('RungeKuttaTrackerPropagator'),
    MultiRecHitUpdator = cms.string('SiTrackerMultiRecHitUpdator'),
    ComponentName = cms.string('groupedMultiRecHitCollector'),
    propagatorOpposite = cms.string('OppositeRungeKuttaTrackerPropagator'),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('RelaxedChi2'),
    Mode = cms.string('Grouped'),
    Debug = cms.bool(False)
)


