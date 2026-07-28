import FWCore.ParameterSet.Config as cms

# Track associations
from Validation.RecoTrack.associators_cff import hltTPClusterProducer, hltTrackAssociatorByHits, tpToHLTpixelTrackAssociation

# Vertex associators
from SimTracker.VertexAssociation.vertexAssociatorByPositionAndTracksProducer_cfi import vertexAssociatorByPositionAndTracksProducer as _VertexAssociatorByPositionAndTracks
from SimTracker.VertexAssociation.secondaryVertexAssociatorByPositionAndTracks_cfi import secondaryVertexAssociatorByPositionAndTracksCPC as _SecondaryVertexAssociatorByPositionAndTracks

# Vertex association producers
from SimTracker.VertexAssociation.vertexCompositePtrCandidateAssociatorEDProducer_cfi import vertexCompositePtrCandidateAssociatorEDProducer as _VertexAssociationCPC

# -------------- PVs from hltGeneralTracks ---------------------------------------------------------------------------
tpToHLTGeneralTrackAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = "hltGeneralTracks"
)
hltPVAssociatorByPositionAndTracks4GeneralTracks = _VertexAssociatorByPositionAndTracks.clone(
    trackAssociations = ["tpToHLTGeneralTrackAssociation"],
    sharedTrackFraction = 0.5, # requires optimization
    weightMethod = "dzError",
    sigmaZ = 10e6
)

# -------------- PVs from hltPixelTracks -----------------------------------------------------------------------------
hltPVAssociatorByPositionAndTracks4PixelTracks = _VertexAssociatorByPositionAndTracks.clone(
    trackAssociations = ["tpToHLTpixelTrackAssociation"],
    sharedTrackFraction = -1, # requires optimization
    weightMethod = "dzError",
    sigmaZ = 10e6
)

# -------------- PVs from hltPFMuonMerging ---------------------------------------------------------------------------
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
hltPVAssociatorByPositionAndTracks4pfMuonMergingTracks = _VertexAssociatorByPositionAndTracks.clone(
    trackAssociations = ["tpToHLTpfMuonMergingTrackAssociation"]
)

# -------------- SVs from hltGeneralTracks ---------------------------------------------------------------------------
hltSVAssociatorByPositionAndTracks4GeneralTracks = _SecondaryVertexAssociatorByPositionAndTracks.clone(
    trackAssociations = ["tpToHLTGeneralTrackAssociation"]
)
hltSVAssociation = _VertexAssociationCPC.clone(
    recoVertices = cms.InputTag("hltDeepInclusiveMergedVerticesPF"),
    simVertices = cms.InputTag("mix", "MergedTrackTruth"),
    associator = cms.InputTag("hltSVAssociatorByPositionAndTracks4GeneralTracks"),
)


# --------------------------------------------------------------------------------------------------------------------
#   Association Tasks for PV and SV validation
# --------------------------------------------------------------------------------------------------------------------

# PV validation association task
hltPVAssociationsTask = cms.Task(
    hltOtherTPClusterProducer,
    hltTrackAssociatorByHits,
    hltOtherTrackAssociatorByHits,
    tpToHLTpixelTrackAssociation,
    hltPVAssociatorByPositionAndTracks4PixelTracks,
    tpToHLTpfMuonMergingTrackAssociation,
    hltPVAssociatorByPositionAndTracks4pfMuonMergingTracks,
    tpToHLTGeneralTrackAssociation,
    hltPVAssociatorByPositionAndTracks4GeneralTracks,
)

# SV validation association task
hltSVAssociationsTask = cms.Task(
    tpToHLTGeneralTrackAssociation,
    hltSVAssociatorByPositionAndTracks4GeneralTracks,
    hltSVAssociation,
)
