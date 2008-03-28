import FWCore.ParameterSet.Config as cms

import copy
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
# Chi2MeasurementEstimatorESProducer
RelaxedChi2Simple = copy.deepcopy(Chi2MeasurementEstimator)
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
RelaxedChi2Simple.ComponentName = 'RelaxedChi2Simple'
RelaxedChi2Simple.MaxChi2 = 100.

