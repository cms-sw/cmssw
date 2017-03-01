import FWCore.ParameterSet.Config as cms

import Validation.RecoTrack.PostProcessorTracker_cfi as _PostProcessorTracker_cfi

postProcessorHLTtracking = _PostProcessorTracker_cfi.postProcessorTrack.clone(
    subDirs = ["HLT/Tracking/ValidationWRTtp/*"]
)

postProcessorHLTtrackingSummary = _PostProcessorTracker_cfi.postProcessorTrackSummary.clone(
    subDirs = ["HLT/Tracking/ValidationWRTtp"]
)

postProcessorHLTtrackingSequence = (
    postProcessorHLTtracking +
    postProcessorHLTtrackingSummary
)
