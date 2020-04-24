import FWCore.ParameterSet.Config as cms

#
# Looser chi2 estimator for electron trajectory building
#   (definition should be moved?)
#
import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
electronChi2 = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone()
electronChi2.ComponentName = cms.string('electronChi2')
electronChi2.nSigma = 3.0
electronChi2.MaxChi2 = 100.0


