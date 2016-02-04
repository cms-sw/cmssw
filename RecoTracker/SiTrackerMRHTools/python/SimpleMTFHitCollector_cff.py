import FWCore.ParameterSet.Config as cms

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
RelaxedChi2Simple = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone()
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import *
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
from RecoTracker.SiTrackerMRHTools.SiTrackerMultiRecHitUpdatorMTF_cff import *
from RecoTracker.SiTrackerMRHTools.SimpleMTFHitCollector_cfi import *
RelaxedChi2Simple.ComponentName = 'RelaxedChi2Simple'
RelaxedChi2Simple.MaxChi2 = 100.

