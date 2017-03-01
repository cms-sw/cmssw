import FWCore.ParameterSet.Config as cms

siTrackerMultiRecHitUpdator = cms.ESProducer("SiTrackerMultiRecHitUpdatorESProducer",
    ComponentName = cms.string('SiTrackerMultiRecHitUpdator'),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    HitPropagator = cms.string('trackingRecHitPropagator'),
    #AnnealingProgram = cms.vdouble(80.0, 9.0, 4.0, 1.0, 1.0,  1.0),
    AnnealingProgram = cms.vdouble(30.0, 18.0, 14.0, 11.0, 6.0, 4.0, 2.0, 1.0),
    ChiSquareCut1D = cms.double(10.8276),
    ChiSquareCut2D = cms.double(13.8155),
    Debug = cms.bool(False)
)


