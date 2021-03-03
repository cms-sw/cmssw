import FWCore.ParameterSet.Config as cms

#import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
CkfElectronCandidates = ckfTrackCandidates.clone(
    TrajectoryBuilderPSet = dict(
	refToPSet_ = 'CkfElectronTrajectoryBuilder'
    )
)
