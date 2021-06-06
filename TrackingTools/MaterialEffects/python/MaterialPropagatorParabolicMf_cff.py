import FWCore.ParameterSet.Config as cms

from TrackingTools.MaterialEffects.MaterialPropagator_cfi import MaterialPropagator
MaterialPropagatorParabolicMF = MaterialPropagator.clone(
    SimpleMagneticField = 'ParabolicMf',
    ComponentName       = 'PropagatorWithMaterialParabolicMf'
)

from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import OppositeMaterialPropagator
OppositeMaterialPropagatorParabolicMF = OppositeMaterialPropagator.clone(
    SimpleMagneticField = 'ParabolicMf',
    ComponentName       = 'PropagatorWithMaterialParabolicMfOpposite'
)
