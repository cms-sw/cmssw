import FWCore.ParameterSet.Config as cms

siTrackerMultiRecHitUpdator = cms.ESProducer("SiTrackerMultiRecHitUpdatorESProducer",
    ComponentName = cms.string('SiTrackerMultiRecHitUpdator'),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    HitPropagator = cms.string('trackingRecHitPropagator'),
    #AnnealingProgram = cms.vdouble(80.0, 9.0, 4.0, 1.0, 1.0,  1.0),
    AnnealingProgram = cms.vdouble(50.0, 30.0, 18.0, 11.0, 6.0, 4.0, 2.0, 1.0),
    ChiSquareCut =cms.double(15.0),
    Debug = cms.bool(False)
)


