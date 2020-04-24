import FWCore.ParameterSet.Config as cms

import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
CkfElectronCandidates = copy.deepcopy(ckfTrackCandidates)
CkfElectronCandidates.TrajectoryBuilderPSet.refToPSet_ = 'CkfElectronTrajectoryBuilder'

