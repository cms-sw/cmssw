import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi import *

from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
from TrackingTools.GeomPropagators.SmartPropagator_cff import *

from RecoMuon.TransientTrackingRecHit.MuonTransientTrackingRecHitBuilder_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *

from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *

Chi2EstimatorForRefit = Chi2MeasurementEstimator.clone()
Chi2EstimatorForRefit.ComponentName = cms.string('Chi2EstimatorForRefit')
Chi2EstimatorForRefit.MaxChi2 = cms.double(100000.0)
Chi2EstimatorForRefit.nSigma = cms.double(3.0)


from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
from TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi import *

KFFitterForRefitOutsideIn = KFTrajectoryFitter.clone()
KFFitterForRefitOutsideIn.ComponentName = cms.string('KFFitterForRefitOutsideIn')
KFFitterForRefitOutsideIn.Propagator = cms.string('SmartPropagatorAnyRKOpposite')
KFFitterForRefitOutsideIn.Updator = cms.string('KFUpdator')
KFFitterForRefitOutsideIn.Estimator = cms.string('Chi2EstimatorForRefit')
KFFitterForRefitOutsideIn.minHits = cms.int32(3)


KFSmootherForRefitOutsideIn = KFTrajectorySmoother.clone()
KFSmootherForRefitOutsideIn.ComponentName = cms.string('KFSmootherForRefitOutsideIn')
KFSmootherForRefitOutsideIn.Propagator = cms.string('SmartPropagatorAnyRKOpposite')
KFSmootherForRefitOutsideIn.Updator = cms.string('KFUpdator')
KFSmootherForRefitOutsideIn.Estimator = cms.string('Chi2EstimatorForRefit')
KFSmootherForRefitOutsideIn.errorRescaling = cms.double(100.0)
KFSmootherForRefitOutsideIn.minHits = cms.int32(3)


#
KFFitterForRefitInsideOut = KFTrajectoryFitter.clone()
KFFitterForRefitInsideOut.ComponentName = cms.string('KFFitterForRefitInsideOut')
KFFitterForRefitInsideOut.Propagator = cms.string('SmartPropagatorAnyRK')
KFFitterForRefitInsideOut.Updator = cms.string('KFUpdator')
KFFitterForRefitInsideOut.Estimator = cms.string('Chi2EstimatorForRefit')
KFFitterForRefitInsideOut.minHits = cms.int32(3)


KFSmootherForRefitInsideOut = KFTrajectorySmoother.clone()
KFSmootherForRefitInsideOut.ComponentName = cms.string('KFSmootherForRefitInsideOut')
KFSmootherForRefitInsideOut.Propagator = cms.string('SmartPropagatorAnyRK')
KFSmootherForRefitInsideOut.Updator = cms.string('KFUpdator')
KFSmootherForRefitInsideOut.Estimator = cms.string('Chi2EstimatorForRefit')
KFSmootherForRefitInsideOut.errorRescaling = cms.double(100.0)
KFSmootherForRefitInsideOut.minHits = cms.int32(3)




