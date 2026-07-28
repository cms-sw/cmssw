import FWCore.ParameterSet.Config as cms

# Track associations
from Validation.RecoTrack.associators_cff import hltTPClusterProducer, hltTrackAssociatorByHits, tpToHLTpixelTrackAssociation

# Vertex associators
from SimTracker.VertexAssociation.vertexAssociatorByPositionAndTracksProducer_cfi import vertexAssociatorByPositionAndTracksProducer as _VertexAssociatorByPositionAndTracks


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

# --------------------------------------------------------------------------------------------------------------------
#   Association Tasks for PV validation
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
