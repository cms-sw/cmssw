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
from RecoJets.JetProducers.caloJetsForTrk_cff import *

unsortedOfflinePrimaryVertices=offlinePrimaryVertices.clone()
offlinePrimaryVertices=sortedPrimaryVertices.clone(vertices="unsortedOfflinePrimaryVertices", particles="trackRefsForJetsBeforeSorting")
offlinePrimaryVerticesWithBS=sortedPrimaryVertices.clone(vertices=cms.InputTag("unsortedOfflinePrimaryVertices","WithBS"), particles="trackRefsForJetsBeforeSorting")
trackWithVertexRefSelectorBeforeSorting = trackWithVertexRefSelector.clone(vertexTag="unsortedOfflinePrimaryVertices")
trackWithVertexRefSelectorBeforeSorting.ptMax=9e99
trackWithVertexRefSelectorBeforeSorting.ptErrorCut=9e99
trackRefsForJetsBeforeSorting = trackRefsForJets.clone(src="trackWithVertexRefSelectorBeforeSorting")


vertexreco = cms.Sequence(unsortedOfflinePrimaryVertices*
                          trackWithVertexRefSelectorBeforeSorting*
                          trackRefsForJetsBeforeSorting*
                          caloJetsForTrk * 
                          offlinePrimaryVertices*
                          offlinePrimaryVerticesWithBS*
                          generalV0Candidates*
                          inclusiveVertexing
                          )

#timing
from RecoVertex.PrimaryVertexProducer.TkClusParameters_cff import DA2DParameters
unsortedOfflinePrimaryVertices1D = unsortedOfflinePrimaryVertices.clone()
unsortedOfflinePrimaryVertices1D.TkFilterParameters.minPt = cms.double(0.7)
offlinePrimaryVertices1D=sortedPrimaryVertices.clone(vertices="unsortedOfflinePrimaryVertices1D", particles="trackRefsForJetsBeforeSorting")
offlinePrimaryVertices1DWithBS=sortedPrimaryVertices.clone(vertices="unsortedOfflinePrimaryVertices1D:WithBS", particles="trackRefsForJetsBeforeSorting")
DA2DParameters.TkDAClusParameters.verbose = cms.untracked.bool(False)
unsortedOfflinePrimaryVertices4D = unsortedOfflinePrimaryVertices.clone( verbose = cms.untracked.bool(False),
                                                                         TkClusParameters = DA2DParameters )
unsortedOfflinePrimaryVertices4D.TkFilterParameters.minPt = cms.double(0.7)
unsortedOfflinePrimaryVertices4D.TrackTimesLabel = cms.InputTag("trackTimeValueMapProducer:generalTracksConfigurableFlatResolutionModel")
unsortedOfflinePrimaryVertices4D.TrackTimeResosLabel = cms.InputTag("trackTimeValueMapProducer:generalTracksConfigurableFlatResolutionModelResolution")
offlinePrimaryVertices4D=sortedPrimaryVertices.clone(vertices="unsortedOfflinePrimaryVertices4D", particles="trackRefsForJetsBeforeSorting")
offlinePrimaryVertices4DWithBS=sortedPrimaryVertices.clone(vertices="unsortedOfflinePrimaryVertices4D:WithBS", particles="trackRefsForJetsBeforeSorting")

from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import tpClusterProducer
from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import quickTrackAssociatorByHits
from SimTracker.TrackAssociation.trackTimeValueMapProducer_cfi import trackTimeValueMapProducer
_phase2_tktiming_vertexreco = cms.Sequence( vertexreco.copy() *
                                            tpClusterProducer *
                                            quickTrackAssociatorByHits *
                                            trackTimeValueMapProducer *
                                            unsortedOfflinePrimaryVertices1D *
                                            offlinePrimaryVertices1D *
                                            offlinePrimaryVertices1DWithBS *
                                            unsortedOfflinePrimaryVertices4D *
                                            offlinePrimaryVertices4D *
                                            offlinePrimaryVertices4DWithBS 
                                            )

from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toReplaceWith(vertexreco, _phase2_tktiming_vertexreco)

