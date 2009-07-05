import FWCore.ParameterSet.Config as cms

GsfTrajectorySmoother = cms.ESProducer("GsfTrajectorySmootherESProducer",
    Merger = cms.string('CloseComponentsMerger5D'),
    ComponentName = cms.string('GsfTrajectorySmoother'),
    MaterialEffectsUpdator = cms.string('ElectronMaterialEffects'),
    ErrorRescaling = cms.double(100.0),
    GeometricalPropagator = cms.string('bwdAnalyticalPropagator'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry')                                
)


