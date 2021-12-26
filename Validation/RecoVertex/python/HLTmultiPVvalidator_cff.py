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


hltPVanalysis = hltMultiPVanalysis.clone(
    trackAssociatorMap = "tpToHLTpfMuonMergingTrackAssociation",
    vertexAssociator   = "vertexAssociatorByPositionAndTracks4pfMuonMergingTracks",
    vertexRecoCollections   = (
    "hltVerticesPFFilter",
    #"hltFastPVPixelVertices"
    )
)
hltMultiPVAssociations = cms.Task(
    hltTrackAssociatorByHits,
    tpToHLTpixelTrackAssociation,
    vertexAssociatorByPositionAndTracks4pixelTracks,
    tpToHLTpfMuonMergingTrackAssociation,
    vertexAssociatorByPositionAndTracks4pfMuonMergingTracks
)

hltMultiPVValidation = cms.Sequence( 
    hltPixelPVanalysis
    + hltPVanalysis,
    hltMultiPVAssociations
)
