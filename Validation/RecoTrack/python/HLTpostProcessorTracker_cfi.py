import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.PostProcessorTracker_cfi import postProcessorTrack as _postProcessorTrack

postProcessorHLTtracking = _postProcessorTrack.clone(
    subDirs = ["HLT/Tracking/ValidationWRTtp/*"]
)
