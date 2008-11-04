import FWCore.ParameterSet.Config as cms

import TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi
MRHFittingSmoother = TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi.KFFittingSmoother.clone()
MRHFittingSmoother.ComponentName = 'MRHFittingSmoother'
MRHFittingSmoother.Fitter = 'MRHFitter'
MRHFittingSmoother.Smoother = 'MRHSmoother'

