import FWCore.ParameterSet.Config as cms

siTrackerMultiRecHitUpdatorMTF = cms.ESProducer("SiTrackerMultiRecHitUpdatorMTFESProducer",
    ComponentName = cms.string('SiTrackerMultiRecHitUpdatorMTF'),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    HitPropagator = cms.string('trackingRecHitPropagator'),
    AnnealingProgram = cms.vdouble(80.0, 9.0, 4.0, 1.0, 1.0, 1.0),
    ChiSquareCut =cms.double(15.0)
)


