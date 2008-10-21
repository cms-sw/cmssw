import FWCore.ParameterSet.Config as cms

from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
from RecoMuon.TransientTrackingRecHit.MuonTransientTrackingRecHitBuilder_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from TrackingTools.GeomPropagators.SmartPropagator_cff import *
Chi2EstimatorForRefit = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('Chi2EstimatorForRefit'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(100000.0)
)

KFFitterForRefitOutsideIn = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('KFFitterForRefitOutsideIn'),
    Estimator = cms.string('Chi2EstimatorForRefit'),
    Propagator = cms.string('SmartPropagatorAnyRK'),
    Updator = cms.string('KFUpdator'),
    minHits = cms.int32(3)
)

KFSmootherForRefitOutsideIn = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(100.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('KFSmootherForRefitOutsideIn'),
    Estimator = cms.string('Chi2EstimatorForRefit'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('SmartPropagatorRK')
)

#
KFFitterForRefitInsideOut = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('KFFitterForRefitInsideOut'),
    Estimator = cms.string('Chi2EstimatorForRefit'),
    Propagator = cms.string('SmartPropagatorAnyRK'),
    Updator = cms.string('KFUpdator'),
    minHits = cms.int32(3)
)

KFSmootherForRefitInsideOut = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(100.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('KFSmootherForRefitInsideOut'),
    Estimator = cms.string('Chi2EstimatorForRefit'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('SmartPropagatorAnyRK')
)


