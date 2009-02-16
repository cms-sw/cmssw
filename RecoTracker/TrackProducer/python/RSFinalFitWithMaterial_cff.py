import FWCore.ParameterSet.Config as cms

from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
from TrackingTools.TrackFitters.RungeKuttaKFTrajectoryFitterESProducer_cfi import *
from TrackingTools.TrackFitters.RungeKuttaKFTrajectorySmootherESProducer_cfi import *
from TrackingTools.TrackFitters.RungeKuttaKFFittingSmootherESProducer_cfi import *
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *

import RecoTracker.TrackProducer.TrackProducer_cfi
rsWithMaterialTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'rsTrackCandidates',
    Fitter = 'RKFittingSmoother',
    AlgorithmName = 'rs'
    )

