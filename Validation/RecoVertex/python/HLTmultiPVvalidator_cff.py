import FWCore.ParameterSet.Config as cms

from Validation.RecoVertex.PrimaryVertexAnalyzer4PUSlimmed_cfi import *

hltMultiPVanalysis = vertexAnalysis.clone()
hltMultiPVanalysis.do_generic_sim_plots  = False
hltMultiPVanalysis.verbose               = cms.untracked.bool(False)
hltMultiPVanalysis.root_folder           = cms.untracked.string("HLT/Vertexing/ValidationWRTsim")
hltMultiPVanalysis.vertexRecoCollections = cms.VInputTag( )
hltMultiPVanalysis.trackAssociatorMap    = cms.untracked.InputTag("trackingParticleRecoTrackAsssociation")
hltMultiPVanalysis.vertexAssociator      = cms.untracked.InputTag("VertexAssociatorByPositionAndTracks")

from Validation.RecoTrack.associators_cff import hltTrackAssociatorByHits, tpToHLTpixelTrackAssociation
from SimTracker.VertexAssociation.VertexAssociatorByPositionAndTracks_cfi import VertexAssociatorByPositionAndTracks as _VertexAssociatorByPositionAndTracks
vertexAssociatorByPositionAndTracks4pixelTracks = _VertexAssociatorByPositionAndTracks.clone()
vertexAssociatorByPositionAndTracks4pixelTracks.trackAssociation = cms.InputTag("tpToHLTpixelTrackAssociation")

tpToHLTpfMuonMergingTrackAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = "hltPFMuonMerging"
)
vertexAssociatorByPositionAndTracks4pfMuonMergingTracks = _VertexAssociatorByPositionAndTracks.clone()
vertexAssociatorByPositionAndTracks4pfMuonMergingTracks.trackAssociation = cms.InputTag("tpToHLTpfMuonMergingTrackAssociation")



hltPixelPVanalysis = hltMultiPVanalysis.clone()
hltPixelPVanalysis.do_generic_sim_plots  = True
hltPixelPVanalysis.trackAssociatorMap    = cms.untracked.InputTag("tpToHLTpixelTrackAssociation")
hltPixelPVanalysis.vertexAssociator      = cms.untracked.InputTag("vertexAssociatorByPositionAndTracks4pixelTracks")
hltPixelPVanalysis.vertexRecoCollections = cms.VInputTag(
    "hltPixelVertices",
    "hltTrimmedPixelVertices",
)



hltPVanalysis = hltMultiPVanalysis.clone()
hltPVanalysis.trackAssociatorMap = cms.untracked.InputTag("tpToHLTpfMuonMergingTrackAssociation")
hltPVanalysis.vertexAssociator   = cms.untracked.InputTag("vertexAssociatorByPositionAndTracks4pfMuonMergingTracks")
hltPVanalysis.vertexRecoCollections   = cms.VInputTag(
    "hltVerticesPFFilter"
#    "hltFastPVPixelVertices"
)

hltMultiPVAssociations = cms.Sequence(
    hltTrackAssociatorByHits +
    tpToHLTpixelTrackAssociation +
    vertexAssociatorByPositionAndTracks4pixelTracks +
    tpToHLTpfMuonMergingTrackAssociation +
    vertexAssociatorByPositionAndTracks4pfMuonMergingTracks
)

hltMultiPVValidation = cms.Sequence( 
    hltMultiPVAssociations +
    hltPixelPVanalysis
    + hltPVanalysis
)
