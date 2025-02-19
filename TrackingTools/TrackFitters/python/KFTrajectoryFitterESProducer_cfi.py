import FWCore.ParameterSet.Config as cms

KFTrajectoryFitter = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('KFFitter'),
    Estimator = cms.string('Chi2'),
    Propagator = cms.string('PropagatorWithMaterial'),
    Updator = cms.string('KFUpdator'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),                                
    minHits = cms.int32(3)
)


