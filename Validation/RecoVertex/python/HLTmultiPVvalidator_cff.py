import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.associators_cff import *

from SimTracker.VertexAssociation.VertexAssociatorByPositionAndTracks_cfi import *
hltVertexAssociatorByPositionAndTracks = VertexAssociatorByPositionAndTracks.clone()
hltVertexAssociatorByPositionAndTracks.trackAssociation = "tpToHLTpixelTrackAssociation"


from Validation.RecoVertex.PrimaryVertexAnalyzer4PUSlimmed_cfi import *

hltMultiPVanalysis = vertexAnalysis.clone()
hltMultiPVanalysis.verbose               = False
hltMultiPVanalysis.sigma_z_match         = 3.0
hltMultiPVanalysis.root_folder           = "HLT/Vertexing/ValidationWRTsim"
hltMultiPVanalysis.recoTrackProducer     = "hltPixelTracks"
hltMultiPVanalysis.trackAssociatorMap    = "tpToHLTpixelTrackAssociation"
hltMultiPVanalysis.vertexAssociator      = "hltVertexAssociatorByPositionAndTracks"
hltMultiPVanalysis.vertexRecoCollections = cms.VInputTag(
    "hltPixelVertices",
    "hltTrimmedPixelVertices"
#    "hltFastPVPixelVertices"
)

hltMultiPVValidation = cms.Sequence( 
    hltTrackAssociatorByHits
    + tpToHLTpixelTrackAssociation
    + hltVertexAssociatorByPositionAndTracks
    + hltMultiPVanalysis
)
