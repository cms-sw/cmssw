import FWCore.ParameterSet.Config as cms

siTrackerMultiRecHitUpdator = cms.ESProducer("SiTrackerMultiRecHitUpdatorESProducer",
    ComponentName = cms.string('SiTrackerMultiRecHitUpdator'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    HitPropagator = cms.string('trackingRecHitPropagator'),
    AnnealingProgram = cms.vdouble(80.0, 9.0, 4.0, 1.0, 1.0, 
        1.0)
)


