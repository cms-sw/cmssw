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
from TrackingTools.TrackFitters.RungeKuttaKFTrajectoryFitterESProducer_cfi import *
# KFTrajectorySmootherESProducer
from TrackingTools.TrackFitters.RungeKuttaKFTrajectorySmootherESProducer_cfi import *
# KFFittingSmootherESProducer
from TrackingTools.TrackFitters.RungeKuttaKFFittingSmootherESProducer_cfi import *
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

import RecoTracker.TrackProducer.TrackProducer_cfi

# include TrackProducer and clone with new module label
cosmictrackfinderP5 = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone()

cosmictrackfinderP5.src = 'cosmicCandidateFinderP5'
cosmictrackfinderP5.TTRHBuilder = 'WithTrackAngle'
cosmictrackfinderP5.AlgorithmName = 'cosmic'
cosmictrackfinderP5.Fitter = 'RKFittingSmoother'

