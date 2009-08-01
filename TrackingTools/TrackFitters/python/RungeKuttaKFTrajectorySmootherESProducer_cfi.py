import FWCore.ParameterSet.Config as cms

RKTrajectorySmoother = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(100.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('RKSmoother'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)


