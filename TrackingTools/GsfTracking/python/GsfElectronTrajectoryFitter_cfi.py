import FWCore.ParameterSet.Config as cms

GsfTrajectoryFitter = cms.ESProducer("GsfTrajectoryFitterESProducer",
    Merger = cms.string('CloseComponentsMerger5D'),
    ComponentName = cms.string('GsfTrajectoryFitter'),
    MaterialEffectsUpdator = cms.string('ElectronMaterialEffects'),
    GeometricalPropagator = cms.string('fwdAnalyticalPropagator'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry')                                
)


