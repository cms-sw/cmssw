import FWCore.ParameterSet.Config as cms

from Validation.RecoVertex.PrimaryVertexAnalyzer4PUSlimmed_cfi import *

hltMultiPVanalysis = vertexAnalysis.clone(
    do_generic_sim_plots  = False,
    verbose               = False,
    root_folder           = "HLT/Vertexing/ValidationWRTsim",
    vertexRecoCollections = [""],
    trackAssociatorMap    = "trackingParticleRecoTrackAsssociation",
    vertexAssociator      = "VertexAssociatorByPositionAndTracks"
)
from Validation.RecoTrack.associators_cff import hltTrackAssociatorByHits, tpToHLTpixelTrackAssociation
from SimTracker.VertexAssociation.VertexAssociatorByPositionAndTracks_cfi import VertexAssociatorByPositionAndTracks as _VertexAssociatorByPositionAndTracks
vertexAssociatorByPositionAndTracks4pixelTracks = _VertexAssociatorByPositionAndTracks.clone(
    trackAssociation = "tpToHLTpixelTrackAssociation"
)
tpToHLTpfMuonMergingTrackAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = "hltPFMuonMerging"
)
vertexAssociatorByPositionAndTracks4pfMuonMergingTracks = _VertexAssociatorByPositionAndTracks.clone(
    trackAssociation = "tpToHLTpfMuonMergingTrackAssociation"
)

hltPixelPVanalysis = hltMultiPVanalysis.clone(
    do_generic_sim_plots  = True,
    trackAssociatorMap    = "tpToHLTpixelTrackAssociation",
    vertexAssociator      = "vertexAssociatorByPositionAndTracks4pixelTracks",
    vertexRecoCollections = (
        "hltPixelVertices",
        "hltTrimmedPixelVertices",
    )
)

def _modifyPixelPVanalysisForPhase2(pvanalysis):
    pvanalysis.vertexRecoCollections = ["hltPhase2PixelVertices"]

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(hltPixelPVanalysis, _modifyPixelPVanalysisForPhase2)

hltPVanalysis = hltMultiPVanalysis.clone(
    trackAssociatorMap = "tpToHLTpfMuonMergingTrackAssociation",
    vertexAssociator   = "vertexAssociatorByPositionAndTracks4pfMuonMergingTracks",
    vertexRecoCollections   = (
    "hltVerticesPFFilter",
    #"hltFastPVPixelVertices"
    )
)

tpToHLTphase2TrackAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = "generalTracks::HLT"
)
vertexAssociatorByPositionAndTracks4phase2HLTTracks = _VertexAssociatorByPositionAndTracks.clone(
    trackAssociation = "tpToHLTphase2TrackAssociation"
)

def _modifyFullPVanalysisForPhase2(pvanalysis):
    pvanalysis.vertexRecoCollections = ["offlinePrimaryVertices::HLT"]
    pvanalysis.trackAssociatorMap = "tpToHLTphase2TrackAssociation"
    pvanalysis.vertexAssociator   = "vertexAssociatorByPositionAndTracks4phase2HLTTracks"

phase2_tracker.toModify(hltPVanalysis, _modifyFullPVanalysisForPhase2)

hltMultiPVAssociations = cms.Task(
    hltTrackAssociatorByHits,
    tpToHLTpixelTrackAssociation,
    vertexAssociatorByPositionAndTracks4pixelTracks,
    tpToHLTpfMuonMergingTrackAssociation,
    vertexAssociatorByPositionAndTracks4pfMuonMergingTracks,
    tpToHLTphase2TrackAssociation,
    vertexAssociatorByPositionAndTracks4phase2HLTTracks
)

hltMultiPVValidation = cms.Sequence( 
    hltPixelPVanalysis
    + hltPVanalysis,
    hltMultiPVAssociations
)
