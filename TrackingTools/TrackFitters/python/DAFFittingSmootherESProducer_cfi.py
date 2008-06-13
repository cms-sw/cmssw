import FWCore.ParameterSet.Config as cms

import TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi
DAFFittingSmoother = TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi.KFFittingSmoother.clone()
DAFFittingSmoother.ComponentName = 'DAFFittingSmoother'
DAFFittingSmoother.Fitter = 'DAFFitter'
DAFFittingSmoother.Smoother = 'DAFSmoother'

