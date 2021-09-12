import FWCore.ParameterSet.Config as cms

import TrackingTools.TrackFitters.KFFittingSmoother_cfi
GsfElectronFittingSmoother = TrackingTools.TrackFitters.KFFittingSmoother_cfi.KFFittingSmoother.clone(
    ComponentName = 'GsfElectronFittingSmoother',
    Fitter        = 'GsfTrajectoryFitter',
    Smoother      = 'GsfTrajectorySmoother',
    MinNumberOfHitsHighEta = 3,
    HighEtaSwitch = 2.5
)
