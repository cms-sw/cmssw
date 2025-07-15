import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.HLTmultiTrackValidator_cfi import *
from SimGeneral.TrackingAnalysis.trackingParticleNumberOfLayersProducer_cff import *

hltTrackValidator = hltMultiTrackValidator.clone(
    label = [
        "hltPixelTracks",
        "hltIter0PFlowTrackSelectionHighPurity",
        "hltIter1PFlowTrackSelectionHighPurity",
        "hltIter1Merged",
        "hltIter2PFlowTrackSelectionHighPurity",
        "hltIter2Merged",
        "hltMergedTracks"
    ]
)

hltMultiTrackValidationTask = cms.Task(
    hltTPClusterProducer
    , trackingParticleNumberOfLayersProducer
    , hltTrackAssociatorByHits
)
hltMultiTrackValidation = cms.Sequence(
    hltTrackValidator,
    hltMultiTrackValidationTask
)

def _modifyForRun3(trackvalidator):
    trackvalidator.label = ["hltPixelTracks", "hltIter0PFlowCtfWithMaterialTracks", "hltIter0PFlowTrackSelectionHighPurity", "hltDoubletRecoveryPFlowCtfWithMaterialTracks", "hltDoubletRecoveryPFlowTrackSelectionHighPurity", "hltMergedTracks"]

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(hltTrackValidator, _modifyForRun3)

def _modifyForPhase2(trackvalidator):
    trackvalidator.label = ["hltGeneralTracks", "hltPhase2PixelTracks", "hltInitialStepTrackSelectionHighPurity", "hltHighPtTripletStepTrackSelectionHighPurity"]

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(hltTrackValidator, _modifyForPhase2)

from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
from Configuration.ProcessModifiers.seedingLST_cff import seedingLST

def _modifyForPhase2LSTTracking(trackvalidator):
    trackvalidator.label = ["hltGeneralTracks", "hltPhase2PixelTracks", "hltInitialStepTrackSelectionHighPuritypTTCLST", "hltInitialStepTrackSelectionHighPuritypLSTCLST", "hltInitialStepTracksT5TCLST", "hltHighPtTripletStepTrackSelectionHighPurity"]
(~seedingLST & trackingLST).toModify(hltTrackValidator, _modifyForPhase2LSTTracking)

def _modifyForPhase2LSTSeeding(trackvalidator):
    trackvalidator.label = ["hltGeneralTracks", "hltPhase2PixelTracks", "hltInitialStepTrackSelectionHighPuritypTTCLST", "hltInitialStepTracksT5TCLST", "hltHighPtTripletStepTrackSelectionHighPuritypLSTCLST"]
(seedingLST & trackingLST).toModify(hltTrackValidator, _modifyForPhase2LSTSeeding)

from Configuration.ProcessModifiers.singleIterPatatrack_cff import singleIterPatatrack
def _modifyForSingleIterPatatrack(trackvalidator):
    trackvalidator.label = ["hltGeneralTracks", "hltPhase2PixelTracks", "hltInitialStepTrackSelectionHighPurity"]
(singleIterPatatrack & ~trackingLST & ~seedingLST).toModify(hltTrackValidator, _modifyForSingleIterPatatrack)

def _modifyForSingleIterPatatrackLST(trackvalidator):
    trackvalidator.label = ["hltGeneralTracks", "hltPhase2PixelTracks", "hltInitialStepTrackSelectionHighPuritypTTCLST", "hltInitialStepTrackSelectionHighPuritypLSTCLST", "hltInitialStepTracksT5TCLST"]
(singleIterPatatrack & ~seedingLST & trackingLST).toModify(hltTrackValidator, _modifyForSingleIterPatatrackLST)

def _modifyForSingleIterPatatrackLSTSeeding(trackvalidator):
    trackvalidator.label = ["hltGeneralTracks", "hltPhase2PixelTracks", "hltInitialStepTrackSelectionHighPuritypTTCLST", "hltInitialStepTracksT5TCLST"]
(singleIterPatatrack & seedingLST & trackingLST).toModify(hltTrackValidator, _modifyForSingleIterPatatrackLSTSeeding)
