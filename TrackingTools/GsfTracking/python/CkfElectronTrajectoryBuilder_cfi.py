import FWCore.ParameterSet.Config as cms

#import copy
from RecoTracker.CkfPattern.CkfTrajectoryBuilder_cfi import *
CkfElectronTrajectoryBuilder = CkfTrajectoryBuilder.clone(
    propagatorAlong = 'fwdElectronPropagator',
    propagatorOpposite = 'bwdElectronPropagator',
    estimator = 'electronChi2'
)
