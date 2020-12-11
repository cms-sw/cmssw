import FWCore.ParameterSet.Config as cms

from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi import *
from TrackingTools.MaterialEffects.Propagators_cff import *
from TrackingTools.TrackFitters.TrackFitters_cff import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *


#could it just use the standard KFFittingSmootherWithOutliersRejectionAndRK ??
import TrackingTools.TrackFitters.KFFittingSmoother_cfi
FittingSmootherRKP5 = TrackingTools.TrackFitters.KFFittingSmoother_cfi.KFFittingSmoother.clone(
    ComponentName    = 'FittingSmootherRKP5',
    Fitter           = 'RKFitter',
    Smoother         = 'RKSmoother',
    MinNumberOfHits  = 4, #why is this set to 4??
    EstimateCut      = 20.0,
    BreakTrajWith2ConsecutiveMissing = False
)

import RecoTracker.TrackProducer.TrackProducer_cfi
ctfWithMaterialTracksCosmics = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src                 = 'ckfTrackCandidatesP5',
    Fitter              = 'FittingSmootherRKP5',
    #TTRHBuilder = 'WithTrackAngle',
    AlgorithmName       = 'ctf',
    GeometricInnerState = True
)
