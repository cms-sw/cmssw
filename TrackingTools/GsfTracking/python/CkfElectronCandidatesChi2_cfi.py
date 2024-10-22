import FWCore.ParameterSet.Config as cms

#
# Looser chi2 estimator for electron trajectory building
#   (definition should be moved?)
#
import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
electronChi2 = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = 'electronChi2',
    nSigma        = 3.0,
    MaxChi2       = 100.0
)
