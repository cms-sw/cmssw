import FWCore.ParameterSet.Config as cms

RKTrajectoryFitter = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('RKFitter'),
    Estimator = cms.string('Chi2'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    Updator = cms.string('KFUpdator'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),                                
    minHits = cms.int32(3)
)


