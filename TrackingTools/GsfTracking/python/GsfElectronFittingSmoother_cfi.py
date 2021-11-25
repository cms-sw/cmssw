import FWCore.ParameterSet.Config as cms

import TrackingTools.TrackFitters.KFFittingSmoother_cfi
GsfElectronFittingSmoother = TrackingTools.TrackFitters.KFFittingSmoother_cfi.KFFittingSmoother.clone(
    ComponentName = 'GsfElectronFittingSmoother',
    Fitter        = 'GsfTrajectoryFitter',
    Smoother      = 'GsfTrajectorySmoother',
    MinNumberOfHitsHighEta = 3,
    HighEtaSwitch = 2.5
)

# Phase2 has extended outer-tracker coverage 
# so no need to relax cuts on number of hits at high eta  
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(GsfElectronFittingSmoother, 
    MinNumberOfHitsHighEta = 5,
    HighEtaSwitch = 5.0
)
