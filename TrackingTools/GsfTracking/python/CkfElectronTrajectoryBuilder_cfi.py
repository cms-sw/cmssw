import FWCore.ParameterSet.Config as cms

import copy
from RecoTracker.CkfPattern.CkfTrajectoryBuilder_cfi import *
CkfElectronTrajectoryBuilder = copy.deepcopy(CkfTrajectoryBuilder)
CkfElectronTrajectoryBuilder.propagatorAlong = 'fwdElectronPropagator'
CkfElectronTrajectoryBuilder.propagatorOpposite = 'bwdElectronPropagator'
CkfElectronTrajectoryBuilder.estimator = 'electronChi2'

