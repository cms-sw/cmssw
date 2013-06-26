import FWCore.ParameterSet.Config as cms

KFTrajectorySmoother = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(100.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('KFSmoother'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFUpdator'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),                                
    Propagator = cms.string('PropagatorWithMaterial')
)


