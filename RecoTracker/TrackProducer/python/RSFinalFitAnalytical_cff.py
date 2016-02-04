import FWCore.ParameterSet.Config as cms

# magnetic field
# cms geometry

# tracker geometry
# tracker numbering
# KFUpdatoerESProducer
from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
# Chi2MeasurementEstimatorESProducer
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
# KFTrajectoryFitterESProducer
from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
# KFTrajectorySmootherESProducer
from TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi import *
# KFFittingSmootherESProducer
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
# AnalyticalPropagatorESProducer
from TrackingTools.GeomPropagators.AnalyticalPropagator_cfi import *
# AnalyticalPropagatorESProducer
from TrackingTools.GeomPropagators.OppositeAnalyticalPropagator_cfi import *
# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import *
# stripCPE
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
# pixelCPE
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
#TransientTrackingBuilder
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
# TrackProducer

import RecoTracker.TrackProducer.TrackProducer_cfi

RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'rsTrackCandidates',
    AlgorithmName = 'rs',
    Propagator = 'AnalyticalPropagator',
    Fitter = 'KFFittingSmoother',
    TTRHBuilder = 'WithTrackAngle'
    )
