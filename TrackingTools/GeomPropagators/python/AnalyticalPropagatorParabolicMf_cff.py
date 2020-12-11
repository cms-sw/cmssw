import FWCore.ParameterSet.Config as cms

from TrackingTools.GeomPropagators.AnalyticalPropagator_cfi import AnalyticalPropagator
AnalyticalPropagatorParabolicMF = AnalyticalPropagator.clone(
    SimpleMagneticField = cms.string('ParabolicMf'),
    ComponentName = 'AnalyticalPropagatorParabolicMf'
)

from TrackingTools.GeomPropagators.OppositeAnalyticalPropagator_cfi import OppositeAnalyticalPropagator
OppositeAnalyticalPropagatorParabolicMF = OppositeAnalyticalPropagator.clone(
    SimpleMagneticField = cms.string('ParabolicMf'),
    ComponentName = 'AnalyticalPropagatorParabolicMfOpposite'
)
