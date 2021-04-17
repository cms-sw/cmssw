import FWCore.ParameterSet.Config as cms

from RecoTracker.CkfPattern.CkfTrajectoryBuilder_cfi import *
CkfElectronTrajectoryBuilder = CkfTrajectoryBuilder.clone(
    propagatorAlong = 'fwdElectronPropagator',
    propagatorOpposite = 'bwdElectronPropagator',
    estimator = 'electronChi2'
)
