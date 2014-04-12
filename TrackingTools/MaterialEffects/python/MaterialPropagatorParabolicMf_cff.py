import FWCore.ParameterSet.Config as cms

from TrackingTools.MaterialEffects.MaterialPropagator_cfi import MaterialPropagator
MaterialPropagatorParabolicMF = MaterialPropagator.clone(
    SimpleMagneticField = cms.string('ParabolicMf'),
    ComponentName = cms.string('PropagatorWithMaterialParabolicMf')
)

from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import OppositeMaterialPropagator
OppositeMaterialPropagatorParabolicMF = OppositeMaterialPropagator.clone(
    SimpleMagneticField = cms.string('ParabolicMf'),
    ComponentName = cms.string('PropagatorWithMaterialParabolicMfOpposite')
)

