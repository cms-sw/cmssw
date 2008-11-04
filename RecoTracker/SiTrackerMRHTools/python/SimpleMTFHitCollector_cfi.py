import FWCore.ParameterSet.Config as cms

simpleMTFHitCollector = cms.ESProducer("MultiTrackFilterCollectorESProducer",
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    MultiRecHitUpdator = cms.string('SiTrackerMultiRecHitUpdatorMTF'),
    ComponentName = cms.string('simpleMTFHitCollector'),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('RelaxedChi2Simple'),
    Mode = cms.string('Simple')
)


