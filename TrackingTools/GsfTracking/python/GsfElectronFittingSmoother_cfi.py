import FWCore.ParameterSet.Config as cms

import TrackingTools.TrackFitters.KFFittingSmoother_cfi
GsfElectronFittingSmoother = TrackingTools.TrackFitters.KFFittingSmoother_cfi.KFFittingSmoother.clone()
GsfElectronFittingSmoother.ComponentName = 'GsfElectronFittingSmoother'
GsfElectronFittingSmoother.Fitter = 'GsfTrajectoryFitter'
GsfElectronFittingSmoother.Smoother = 'GsfTrajectorySmoother'


