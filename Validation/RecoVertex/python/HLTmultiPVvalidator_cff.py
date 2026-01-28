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

from Validation.RecoTrack.associators_cff import hltTPClusterProducer, hltTrackAssociatorByHits, tpToHLTpixelTrackAssociation
from SimTracker.VertexAssociation.VertexAssociatorByPositionAndTracks_cfi import VertexAssociatorByPositionAndTracks as _VertexAssociatorByPositionAndTracks
vertexAssociatorByPositionAndTracks4pixelTracks = _VertexAssociatorByPositionAndTracks.clone(
    trackAssociation = "tpToHLTpixelTrackAssociation",
    sharedTrackFraction = -1, # requires optimization
    weightMethod = "dzError",
)
hltOtherTPClusterProducer = hltTPClusterProducer.clone(
    stripClusterOtherSrc = "hltSiStripRawToClustersFacilityOnDemand"
)
hltOtherTrackAssociatorByHits = hltTrackAssociatorByHits.clone(
    cluster2TPSrc = cms.InputTag("hltOtherTPClusterProducer")
)
tpToHLTpfMuonMergingTrackAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = "hltPFMuonMerging",
    associator = cms.InputTag('hltOtherTrackAssociatorByHits')
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

hltPixelPVanalysisReconstructable = hltMultiPVanalysis.clone(
    do_generic_sim_plots  = True,
    use_reconstructable_simvertices = True,
    reco_tracks_for_reconstructable_simvertices = 1, #inclusive, below or equal discard sim vertex.
    root_folder           = "HLT/Vertexing/ValidationWRTReconstructableSim",
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
phase2_tracker.toModify(hltPixelPVanalysisReconstructable, _modifyPixelPVanalysisForPhase2)

hltPVanalysis = hltMultiPVanalysis.clone(
    trackAssociatorMap = "tpToHLTpfMuonMergingTrackAssociation",
    vertexAssociator   = "vertexAssociatorByPositionAndTracks4pfMuonMergingTracks",
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
    vertexAssociator      = "vertexAssociatorByPositionAndTracks4pfMuonMergingTracks",
    vertexRecoCollections = (
        "hltVerticesPFFilter",
    )
)

tpToHLTphase2TrackAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = "hltGeneralTracks"
)
vertexAssociatorByPositionAndTracks4phase2HLTTracks = _VertexAssociatorByPositionAndTracks.clone(
    trackAssociation = "tpToHLTphase2TrackAssociation",
    sharedTrackFraction = 0.5, # requires optimization
    weightMethod = "dzError",
)

def _modifyFullPVanalysisForPhase2(pvanalysis):
    pvanalysis.vertexRecoCollections = ["hltOfflinePrimaryVertices"]
    pvanalysis.trackAssociatorMap = "tpToHLTphase2TrackAssociation"
    pvanalysis.vertexAssociator   = "vertexAssociatorByPositionAndTracks4phase2HLTTracks"

phase2_tracker.toModify(hltPVanalysis, _modifyFullPVanalysisForPhase2)
phase2_tracker.toModify(hltPVanalysisReconstructable, _modifyFullPVanalysisForPhase2)

hltMultiPVAssociations = cms.Task(
    hltOtherTPClusterProducer,
    hltTrackAssociatorByHits,
    hltOtherTrackAssociatorByHits,
    tpToHLTpixelTrackAssociation,
    vertexAssociatorByPositionAndTracks4pixelTracks,
    tpToHLTpfMuonMergingTrackAssociation,
    vertexAssociatorByPositionAndTracks4pfMuonMergingTracks,
    tpToHLTphase2TrackAssociation,
    vertexAssociatorByPositionAndTracks4phase2HLTTracks
)

hltMultiPVValidation = cms.Sequence(hltPixelPVanalysis +
                                    hltPixelPVanalysisReconstructable +
                                    hltPVanalysis +
                                    hltPVanalysisReconstructable,
                                    hltMultiPVAssociations)
