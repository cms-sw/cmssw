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


vertexrecoTask = cms.Task(unsortedOfflinePrimaryVertices,
                          trackWithVertexRefSelectorBeforeSorting,
                          trackRefsForJetsBeforeSorting,
                          offlinePrimaryVertices,
                          offlinePrimaryVerticesWithBS,
                          generalV0Candidates,
                          caloJetsForTrkTask,
                          inclusiveVertexingTask
                          )
vertexreco = cms.Sequence(vertexrecoTask)

#modifications for timing
#from RecoVertex.Configuration.RecoVertex_phase2_timing_cff import _phase2_tktiming_vertexrecoTask, unsortedOfflinePrimaryVertices4D, offlinePrimaryVertices4D, offlinePrimaryVertices4DWithBS, DA2D_vectParameters
from RecoVertex.Configuration.RecoVertex_phase2_timing_cff import *
_phase2_tktiming_vertexrecoTask = cms.Task( vertexrecoTask.copy() ,
                                            tpClusterProducer ,
                                            quickTrackAssociatorByHits ,
                                            trackTimeValueMapProducer ,
                                            unsortedOfflinePrimaryVertices4DnoPID ,
                                            trackWithVertexRefSelectorBeforeSorting4DnoPID ,
                                            trackRefsForJetsBeforeSorting4DnoPID ,
                                            offlinePrimaryVertices4DnoPID ,
                                            offlinePrimaryVertices4DnoPIDWithBS,
                                            tofPID,
                                            unsortedOfflinePrimaryVertices4Dfastsim,
                                            trackWithVertexRefSelectorBeforeSorting4Dfastsim ,
                                            trackRefsForJetsBeforeSorting4Dfastsim ,
                                            offlinePrimaryVertices4Dfastsim,
                                            offlinePrimaryVertices4DfastsimWithBS,
                                            unsortedOfflinePrimaryVertices3D,
                                            trackWithVertexRefSelectorBeforeSorting3D ,
                                            trackRefsForJetsBeforeSorting3D,
                                            offlinePrimaryVertices3D,
                                            offlinePrimaryVertices3DWithBS,
                                            )

from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toReplaceWith(vertexrecoTask, _phase2_tktiming_vertexrecoTask)
phase2_timing.toReplaceWith(unsortedOfflinePrimaryVertices, unsortedOfflinePrimaryVertices4D)
phase2_timing.toReplaceWith(offlinePrimaryVertices, offlinePrimaryVertices4D)
phase2_timing.toReplaceWith(offlinePrimaryVerticesWithBS, offlinePrimaryVertices4DWithBS)
phase2_timing.toModify(offlinePrimaryVertices, vertices = "unsortedOfflinePrimaryVertices", particles = "trackRefsForJetsBeforeSorting")
phase2_timing.toModify(offlinePrimaryVerticesWithBS, vertices = "unsortedOfflinePrimaryVertices:WithBS", particles = "trackRefsForJetsBeforeSorting")
