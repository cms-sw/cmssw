import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.HLTmultiTrackValidator_cfi import *
from SimGeneral.TrackingAnalysis.trackingParticleNumberOfLayersProducer_cff import *
from Validation.RecoTrack.cutsRecoTracks_cfi import cutsRecoTracks as _cutsRecoTracks

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

# Pixel-less track selector
hltPixelLessTracks = _cutsRecoTracks.clone(
    throwOnMissing = cms.bool(False), # HLT collection might be missing
    src = "hltMergedTracks",
    beamSpot = "hltOnlineBeamSpot",
    minLayer = 3,
    maxPixelHit = 0
)

# Tracks with at least one pixel hit
hltWithPixelTracks = _cutsRecoTracks.clone(
    throwOnMissing = cms.bool(False), # HLT collection might be missing
    src = "hltMergedTracks",
    beamSpot = "hltOnlineBeamSpot",
    minLayer = 3,
    minPixelHit = 1
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
    trackvalidator.label = ["hltGeneralTracks", "hltPhase2PixelTracks", "hltInitialStepTrackSelectionHighPurity", "hltHighPtTripletStepTrackSelectionHighPurity", "hltPixelLessTracks", "hltWithPixelTracks"]

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(hltTrackValidator, _modifyForPhase2)
phase2_tracker.toModify(hltPixelLessTracks, src = "hltGeneralTracks")
phase2_tracker.toModify(hltWithPixelTracks, src = "hltGeneralTracks")

## for Phase 2 only (no pixelless tracks in Run3) run the track selectors
phase2_tracker.toReplaceWith(
    hltMultiTrackValidation,
    cms.Sequence(
        hltPixelLessTracks +
        hltWithPixelTracks +
        hltTrackValidator,
        hltMultiTrackValidationTask
    )
)


from DQM.TrackingMonitorSource.TrackToTrackComparisonHists_cfi import TrackToTrackComparisonHists as _TrackToTrackComparisonHists
hltInitialStepTrackToTrackSerialSync = _TrackToTrackComparisonHists.clone(
    requireValidHLTPaths = False,
    ignoreLumiPUPlots = True,
    monitoredTrack = "hltInitialStepTracks",
    monitoredBeamSpot = "hltOnlineBeamSpot",
    monitoredPrimaryVertices = "hltPhase2PixelVertices",
    referenceTrack = "hltInitialStepTracksSerialSync",
    referenceBeamSpot = "hltOnlineBeamSpot",
    referencePrimaryVertices = "hltPhase2PixelVertices",
    topDirName = cms.string('HLT/Tracking/ValidationWRTSerialSync/initialStepTracks'),
)

from Configuration.ProcessModifiers.alpakaValidationLST_cff import alpakaValidationLST
alpakaValidationLST.toReplaceWith(hltMultiTrackValidation, cms.Sequence(
    hltMultiTrackValidation.copy()
    +hltInitialStepTrackToTrackSerialSync
    )
)


from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
from Configuration.ProcessModifiers.singleIterPatatrack_cff import singleIterPatatrack

def _modifyForSingleIterPatatrack(trackvalidator):
    trackvalidator.label = ["hltGeneralTracks", "hltPhase2PixelTracks", "hltInitialStepTrackSelectionHighPurity", "hltPixelLessTracks", "hltWithPixelTracks"]
singleIterPatatrack.toModify(hltTrackValidator, _modifyForSingleIterPatatrack)

def _modifyForNGTScouting(trackvalidator):
    trackvalidator.label = ["hltGeneralTracks", "hltPhase2PixelTracks"]
(ngtScouting & ~trackingLST).toModify(hltTrackValidator, _modifyForNGTScouting)

def _modifyForNGTScoutingLST(trackvalidator):
    trackvalidator.label = ["hltGeneralTracks", "hltPhase2PixelTracks", "hltInitialStepTracksT4T5TCLST", "hltPixelLessTracks", "hltWithPixelTracks"]
(ngtScouting & trackingLST).toModify(hltTrackValidator, _modifyForNGTScoutingLST)
