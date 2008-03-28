import FWCore.ParameterSet.Config as cms

groupedMultiRecHitCollector = cms.ESProducer("MultiRecHitCollectorESProducer",
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    MultiRecHitUpdator = cms.string('SiTrackerMultiRecHitUpdator'),
    ComponentName = cms.string('groupedMultiRecHitCollector'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('RelaxedChi2'),
    Mode = cms.string('Grouped')
)


