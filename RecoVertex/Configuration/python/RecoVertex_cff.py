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
offlinePrimaryVertices=sortedPrimaryVertices.clone(
    vertices="unsortedOfflinePrimaryVertices", 
    particles="trackRefsForJetsBeforeSorting"
)
offlinePrimaryVerticesWithBS=sortedPrimaryVertices.clone(
    vertices="unsortedOfflinePrimaryVertices:WithBS", 
    particles="trackRefsForJetsBeforeSorting"
)
trackWithVertexRefSelectorBeforeSorting = trackWithVertexRefSelector.clone(
    vertexTag="unsortedOfflinePrimaryVertices",
    ptMax=9e99,
    ptErrorCut=9e99
)
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
from RecoVertex.Configuration.RecoVertex_phase2_timing_cff import (tpClusterProducer ,
                                                                  quickTrackAssociatorByHits ,
                                                                  trackTimeValueMapProducer ,
                                                                  unsortedOfflinePrimaryVertices4DnoPID ,
                                                                  trackWithVertexRefSelectorBeforeSorting4DnoPID ,
                                                                  trackRefsForJetsBeforeSorting4DnoPID ,
                                                                  offlinePrimaryVertices4DnoPID ,
                                                                  offlinePrimaryVertices4DnoPIDWithBS,
                                                                  unsortedOfflinePrimaryVertices4DwithPID ,
                                                                  offlinePrimaryVertices4DwithPID ,
                                                                  offlinePrimaryVertices4DwithPIDWithBS,
                                                                  tofPID,
                                                                  tofPID4DnoPID,
                                                                  unsortedOfflinePrimaryVertices4D,
                                                                  trackWithVertexRefSelectorBeforeSorting4D,
                                                                  trackRefsForJetsBeforeSorting4D,
                                                                  offlinePrimaryVertices4D,
                                                                  offlinePrimaryVertices4DWithBS)

_phase2_tktiming_vertexrecoTask = cms.Task( vertexrecoTask.copy() ,
                                            tpClusterProducer ,
                                            quickTrackAssociatorByHits ,
                                            trackTimeValueMapProducer ,
                                            unsortedOfflinePrimaryVertices4D,
                                            trackWithVertexRefSelectorBeforeSorting4D ,
                                            trackRefsForJetsBeforeSorting4D,
                                            offlinePrimaryVertices4D,
                                            offlinePrimaryVertices4DWithBS,
                                            )

_phase2_tktiming_layer_vertexrecoTask = cms.Task( _phase2_tktiming_vertexrecoTask.copy() ,
                                            unsortedOfflinePrimaryVertices4DnoPID ,
                                            trackWithVertexRefSelectorBeforeSorting4DnoPID ,
                                            trackRefsForJetsBeforeSorting4DnoPID ,
                                            offlinePrimaryVertices4DnoPID ,
                                            offlinePrimaryVertices4DnoPIDWithBS,
                                            tofPID,
                                            tofPID4DnoPID,
                                            )

from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toReplaceWith(vertexrecoTask, _phase2_tktiming_vertexrecoTask)

from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
phase2_timing_layer.toReplaceWith(vertexrecoTask, _phase2_tktiming_layer_vertexrecoTask)
phase2_timing_layer.toReplaceWith(unsortedOfflinePrimaryVertices4D, unsortedOfflinePrimaryVertices4DwithPID.clone())
phase2_timing_layer.toReplaceWith(offlinePrimaryVertices4D, offlinePrimaryVertices4DwithPID.clone())
phase2_timing_layer.toReplaceWith(offlinePrimaryVertices4DWithBS, offlinePrimaryVertices4DwithPIDWithBS.clone())
phase2_timing_layer.toModify(offlinePrimaryVertices4D, vertices = "unsortedOfflinePrimaryVertices4D", particles = "trackRefsForJetsBeforeSorting4D")
phase2_timing_layer.toModify(offlinePrimaryVertices4DWithBS, vertices = "unsortedOfflinePrimaryVertices4D:WithBS", particles = "trackRefsForJetsBeforeSorting4D")

