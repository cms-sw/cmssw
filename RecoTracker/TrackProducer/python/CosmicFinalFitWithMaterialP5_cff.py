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
# KFTrajectorySmootherESProducer
# KFFittingSmootherESProducer
from TrackingTools.TrackFitters.RungeKuttaFitters_cff import *
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
cosmictrackfinderCosmics = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone()

cosmictrackfinderCosmics.src = 'cosmicCandidateFinderP5'
cosmictrackfinderCosmics.TTRHBuilder = 'WithTrackAngle'
cosmictrackfinderCosmics.AlgorithmName = 'cosmic'
cosmictrackfinderCosmics.Fitter = 'RKFittingSmoother'

