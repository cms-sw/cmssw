import FWCore.ParameterSet.Config as cms

trackingRecHitPropagator = cms.ESProducer("TrackingRecHitPropagatorESProducer",
    ComponentName = cms.string('trackingRecHitPropagator')
)


