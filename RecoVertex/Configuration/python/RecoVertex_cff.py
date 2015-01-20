import FWCore.ParameterSet.Config as cms

# Reco Vertex
# initialize magnetic field #########################
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import *
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVerticesWithBS_cfi import *
from RecoVertex.V0Producer.generalV0Candidates_cff import *
from RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff import *

from CommonTools.RecoAlgos.TrackWithVertexRefSelector_cfi import *
from RecoJets.JetProducers.TracksForJets_cff import *
from CommonTools.RecoAlgos.sortedPrimaryVertices_cfi import *

unsortedOfflinePrimaryVertices=offlinePrimaryVertices.clone()
offlinePrimaryVertices=sortedPrimaryVertices.clone(vertices="unsortedOfflinePrimaryVertices")
offlinePrimaryVerticesWithBS=sortedPrimaryVertices.clone(vertices=cms.InputTag("unsortedOfflinePrimaryVertices","WithBS"))
trackWithVertexRefSelector.vertexTag="unsortedOfflinePrimaryVertices"

vertexreco = cms.Sequence(unsortedOfflinePrimaryVertices*
        trackWithVertexRefSelector*
        trackRefsForJets*
        offlinePrimaryVertices*
        offlinePrimaryVerticesWithBS*
        generalV0Candidates*
        inclusiveVertexing
)
