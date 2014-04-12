import FWCore.ParameterSet.Config as cms

import copy
from RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi import *
CkfElectronTrajectoryBuilder = copy.deepcopy(CkfTrajectoryBuilder)
CkfElectronTrajectoryBuilder.ComponentName = 'CkfElectronTrajectoryBuilder'
CkfElectronTrajectoryBuilder.propagatorAlong = 'fwdElectronPropagator'
CkfElectronTrajectoryBuilder.propagatorOpposite = 'bwdElectronPropagator'
CkfElectronTrajectoryBuilder.estimator = 'electronChi2'

