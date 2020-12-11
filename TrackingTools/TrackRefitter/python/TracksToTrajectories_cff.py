import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi import *

from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
from TrackingTools.GeomPropagators.SmartPropagator_cff import *

from RecoMTD.TransientTrackingRecHit.MTDTransientTrackingRecHitBuilder_cfi import *
from RecoMuon.TransientTrackingRecHit.MuonTransientTrackingRecHitBuilder_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *

from TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi import *

Chi2EstimatorForRefit = Chi2MeasurementEstimator.clone(
    ComponentName = 'Chi2EstimatorForRefit',
    MaxChi2 = 100000.0,
    nSigma = 3.0
)

from TrackingTools.TrackFitters.KFTrajectoryFitter_cfi import *
from TrackingTools.TrackFitters.KFTrajectorySmoother_cfi import *

KFFitterForRefitOutsideIn = KFTrajectoryFitter.clone(
    ComponentName = 'KFFitterForRefitOutsideIn',
    Propagator = 'SmartPropagatorAnyRKOpposite',
    Updator = 'KFUpdator',
    Estimator = 'Chi2EstimatorForRefit',
    minHits = 3
)

KFSmootherForRefitOutsideIn = KFTrajectorySmoother.clone(
    ComponentName = 'KFSmootherForRefitOutsideIn',
    Propagator = 'SmartPropagatorAnyRKOpposite',
    Updator = 'KFUpdator',
    Estimator = 'Chi2EstimatorForRefit',
    errorRescaling = 100.0,
    minHits = 3
)
#
KFFitterForRefitInsideOut = KFTrajectoryFitter.clone(
    ComponentName = 'KFFitterForRefitInsideOut',
    Propagator = 'SmartPropagatorAnyRK',
    Updator = 'KFUpdator',
    Estimator = 'Chi2EstimatorForRefit',
    minHits = 3
)

KFSmootherForRefitInsideOut = KFTrajectorySmoother.clone(
    ComponentName = 'KFSmootherForRefitInsideOut',
    Propagator = 'SmartPropagatorAnyRK',
    Updator = 'KFUpdator',
    Estimator = 'Chi2EstimatorForRefit',
    errorRescaling = 100.0,
    minHits = 3
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
# FastSim doesn't use Runge Kute for propagation
# the following propagators are not used in FastSim, but just to be sure...
fastSim.toModify(KFFitterForRefitOutsideIn, Propagator = 'SmartPropagatorAny')
fastSim.toModify(KFSmootherForRefitOutsideIn, Propagator = 'SmartPropagator')
fastSim.toModify(KFFitterForRefitInsideOut, Propagator = "SmartPropagatorAny")
fastSim.toModify(KFSmootherForRefitInsideOut, Propagator = "SmartPropagatorAny")
