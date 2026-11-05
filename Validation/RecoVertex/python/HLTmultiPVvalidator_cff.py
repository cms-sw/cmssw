import FWCore.ParameterSet.Config as cms

from Validation.RecoVertex.PrimaryVertexAnalyzer4PUSlimmed_cfi import *
from Validation.RecoVertex.associators_cff import *

hltMultiPVanalysis = vertexAnalysis.clone(
    do_generic_sim_plots  = False,
    verbose               = False,
    root_folder           = "HLT/Vertexing/ValidationWRTsim",
    vertexRecoCollections = [""],
    trackAssociatorMap    = "trackingParticleRecoTrackAsssociation",
    vertexAssociator      = "vertexAssociatorByPositionAndTracksProducer"
)

hltPixelPVanalysis = hltMultiPVanalysis.clone(
    do_generic_sim_plots  = True,
    trackAssociatorMap    = "tpToHLTpixelTrackAssociation",
    vertexAssociator      = "hltPVAssociatorByPositionAndTracks4PixelTracks",
    vertexRecoCollections = (
        "hltPixelVertices",
        "hltTrimmedPixelVertices",
    )
)

hltPixelPVanalysisReconstructable = hltMultiPVanalysis.clone(
    do_generic_sim_plots  = True,
    use_reconstructable_simvertices = True,
    reco_tracks_for_reconstructable_simvertices = 1, #inclusive, below or equal discard sim vertex.
    root_folder           = "HLT/Vertexing/ValidationWRTReconstructableSim",
    trackAssociatorMap    = "tpToHLTpixelTrackAssociation",
    vertexAssociator      = "hltPVAssociatorByPositionAndTracks4PixelTracks",
    vertexRecoCollections = (
        "hltPixelVertices",
        "hltTrimmedPixelVertices",
    )
)

def _modifyPixelPVanalysisForPhase2(pvanalysis):
    pvanalysis.vertexRecoCollections = ["hltPhase2PixelVertices"]

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(hltPixelPVanalysis, _modifyPixelPVanalysisForPhase2)
phase2_tracker.toModify(hltPixelPVanalysisReconstructable, _modifyPixelPVanalysisForPhase2)

hltPVanalysis = hltMultiPVanalysis.clone(
    trackAssociatorMap = "tpToHLTpfMuonMergingTrackAssociation",
    vertexAssociator   = "hltPVAssociatorByPositionAndTracks4pfMuonMergingTracks",
    vertexRecoCollections   = (
    "hltVerticesPFFilter",
    #"hltFastPVPixelVertices"
    )
)

hltPVanalysisReconstructable = hltMultiPVanalysis.clone(
    do_generic_sim_plots  = False, # to not produce fill the ones from hltPixelPVanalysisReconstructable twice
    use_reconstructable_simvertices = True,
    reco_tracks_for_reconstructable_simvertices = 1, #inclusive, below or equal discard sim vertex.
    root_folder           = "HLT/Vertexing/ValidationWRTReconstructableSim",
    trackAssociatorMap    = "tpToHLTpfMuonMergingTrackAssociation",
    vertexAssociator      = "hltPVAssociatorByPositionAndTracks4pfMuonMergingTracks",
    vertexRecoCollections = (
        "hltVerticesPFFilter",
    )
)

def _modifyFullPVanalysisForPhase2(pvanalysis):
    pvanalysis.vertexRecoCollections = ["hltOfflinePrimaryVertices"]
    pvanalysis.trackAssociatorMap = "tpToHLTGeneralTrackAssociation"
    pvanalysis.vertexAssociator   = "hltPVAssociatorByPositionAndTracks4GeneralTracks"

phase2_tracker.toModify(hltPVanalysis, _modifyFullPVanalysisForPhase2)
phase2_tracker.toModify(hltPVanalysisReconstructable, _modifyFullPVanalysisForPhase2)

hltMultiPVValidation = cms.Sequence(hltPixelPVanalysis +
                                    hltPixelPVanalysisReconstructable +
                                    hltPVanalysis +
                                    hltPVanalysisReconstructable,
                                    hltPVAssociationsTask)
