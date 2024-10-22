import FWCore.ParameterSet.Config as cms

from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
CkfElectronCandidates = ckfTrackCandidates.clone(
    TrajectoryBuilderPSet = dict(
	refToPSet_ = 'CkfElectronTrajectoryBuilder'
    )
)
