import FWCore.ParameterSet.Config as cms

from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
CkfElectronCandidates = ckfTrackCandidates.clone(
    TrajectoryBuilderPSet = dict(
	refToPSet_ = 'CkfElectronTrajectoryBuilder'
    )
)
# foo bar baz
# bqOy5xjyB1UJK
# wXlK8DzXnVv6H
