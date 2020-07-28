import FWCore.ParameterSet.Config as cms

from TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi import *
# Chi2MeasurementEstimatorESProducer
RelaxedChi2 = Chi2MeasurementEstimator.clone(
    ComponentName = 'RelaxedChi2',
    MaxChi2       = 100.
)
#replace RelaxedChi2.nSigma = 3.
# MeasurementTracker
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
# MultiRecHitUpdator
from RecoTracker.SiTrackerMRHTools.SiTrackerMultiRecHitUpdator_cff import *
#MultiRecHitCollector
from RecoTracker.SiTrackerMRHTools.GroupedMultiRecHitCollector_cfi import *

