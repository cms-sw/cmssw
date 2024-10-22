import FWCore.ParameterSet.Config as cms

from TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi import *
# Chi2MeasurementEstimatorESProducer
RelaxedChi2Simple = Chi2MeasurementEstimator.clone(
    ComponentName = 'RelaxedChi2Simple',
    MaxChi2       = 100.
)
#replace RelaxedChi2.nSigma = 3.
# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import *
# MeasurementTracker
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
# MultiRecHitUpdator
from RecoTracker.SiTrackerMRHTools.SiTrackerMultiRecHitUpdator_cff import *
#MultiRecHitCollector
from RecoTracker.SiTrackerMRHTools.SimpleMultiRecHitCollector_cfi import *

