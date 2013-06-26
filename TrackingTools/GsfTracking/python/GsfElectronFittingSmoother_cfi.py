import FWCore.ParameterSet.Config as cms

import TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi
GsfElectronFittingSmoother = TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi.KFFittingSmoother.clone()
GsfElectronFittingSmoother.ComponentName = 'GsfElectronFittingSmoother'
GsfElectronFittingSmoother.Fitter = 'GsfTrajectoryFitter'
GsfElectronFittingSmoother.Smoother = 'GsfTrajectorySmoother'


