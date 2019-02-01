import FWCore.ParameterSet.Config as cms
from RecoVertex.Configuration.RecoVertex_cff import unsortedOfflinePrimaryVertices, trackWithVertexRefSelector, trackRefsForJets, sortedPrimaryVertices, offlinePrimaryVertices, offlinePrimaryVerticesWithBS,vertexrecoTask

from RecoVertex.PrimaryVertexProducer.TkClusParameters_cff import DA2D_vectParameters

unsortedOfflinePrimaryVertices4D = unsortedOfflinePrimaryVertices.clone(TkClusParameters = DA2D_vectParameters,
                                                                        TrackTimesLabel = cms.InputTag("trackTimeValueMapProducer","generalTracksConfigurableFlatResolutionModel"),
                                                                        TrackTimeResosLabel = cms.InputTag("trackTimeValueMapProducer","generalTracksConfigurableFlatResolutionModelResolution"),
                                                                        )
trackWithVertexRefSelectorBeforeSorting4D = trackWithVertexRefSelector.clone(vertexTag="unsortedOfflinePrimaryVertices4D",
                                                                                    ptMax=9e99,
                                                                                    ptErrorCut=9e99)
trackRefsForJetsBeforeSorting4D = trackRefsForJets.clone(src="trackWithVertexRefSelectorBeforeSorting4D")
offlinePrimaryVertices4D=sortedPrimaryVertices.clone(vertices="unsortedOfflinePrimaryVertices4D",
                                                            particles="trackRefsForJetsBeforeSorting4D",
                                                            trackTimeTag=cms.InputTag("trackTimeValueMapProducer","generalTracksConfigurableFlatResolutionModel"),
                                                            trackTimeResoTag=cms.InputTag("trackTimeValueMapProducer","generalTracksConfigurableFlatResolutionModelResolution"),
                                                            assignment=dict(useTiming=True))
offlinePrimaryVertices4DWithBS=offlinePrimaryVertices4D.clone(vertices="unsortedOfflinePrimaryVertices4D:WithBS")

unsortedOfflinePrimaryVertices4DnoPID = unsortedOfflinePrimaryVertices4D.clone(TrackTimesLabel = "trackExtenderWithMTD:generalTrackt0",
                                                                         TrackTimeResosLabel = "trackExtenderWithMTD:generalTracksigmat0",
                                                                         )
trackWithVertexRefSelectorBeforeSorting4DnoPID = trackWithVertexRefSelector.clone(vertexTag="unsortedOfflinePrimaryVertices4DnoPID",
                                                                                  ptMax=9e99,
                                                                                  ptErrorCut=9e99)
trackRefsForJetsBeforeSorting4DnoPID = trackRefsForJets.clone(src="trackWithVertexRefSelectorBeforeSorting4DnoPID")
offlinePrimaryVertices4DnoPID=offlinePrimaryVertices4D.clone(vertices="unsortedOfflinePrimaryVertices4DnoPID",
                                                          particles="trackRefsForJetsBeforeSorting4DnoPID",
                                                          trackTimeTag="trackExtenderWithMTD:generalTrackt0",
                                                          trackTimeResoTag="trackExtenderWithMTD:generalTracksigmat0")
offlinePrimaryVertices4DnoPIDWithBS=offlinePrimaryVertices4DnoPID.clone(vertices="unsortedOfflinePrimaryVertices4DnoPID:WithBS")

unsortedOfflinePrimaryVertices4DwithPID = unsortedOfflinePrimaryVertices4D.clone(TrackTimesLabel = "tofPIDfor4DwithPID:t0safe",
                                                                                 TrackTimeResosLabel = "tofPIDfor4DwithPID:sigmat0safe",
                                                                                 )
trackWithVertexRefSelectorBeforeSorting4DwithPID = trackWithVertexRefSelector.clone(vertexTag="unsortedOfflinePrimaryVertices4DwithPID",
                                                                             ptMax=9e99,
                                                                             ptErrorCut=9e99)
trackRefsForJetsBeforeSorting4DwithPID = trackRefsForJets.clone(src="trackWithVertexRefSelectorBeforeSorting4DwithPID")
offlinePrimaryVertices4DwithPID=offlinePrimaryVertices4D.clone(vertices="unsortedOfflinePrimaryVertices4DwithPID",
                                                     particles="trackRefsForJetsBeforeSorting4DwithPID",
                                                     trackTimeTag="tofPIDfor4DwithPID:t0safe",
                                                     trackTimeResoTag="tofPIDfor4DwithPID:sigmat0safe")
offlinePrimaryVertices4DwithPIDWithBS=offlinePrimaryVertices4DwithPID.clone(vertices="unsortedOfflinePrimaryVertices4DwithPID:WithBS")

unsortedOfflinePrimaryVertices4Dfastsim = unsortedOfflinePrimaryVertices4D.clone()
trackWithVertexRefSelectorBeforeSorting4Dfastsim = trackWithVertexRefSelector.clone(vertexTag="unsortedOfflinePrimaryVertices4Dfastsim",
                                                                                    ptMax=9e99,
                                                                                    ptErrorCut=9e99)
trackRefsForJetsBeforeSorting4Dfastsim = trackRefsForJets.clone(src="trackWithVertexRefSelectorBeforeSorting4Dfastsim")
offlinePrimaryVertices4Dfastsim=offlinePrimaryVertices4D.clone(vertices="unsortedOfflinePrimaryVertices4Dfastsim",
                                                            particles="trackRefsForJetsBeforeSorting4Dfastsim")
offlinePrimaryVertices4DfastsimWithBS=offlinePrimaryVertices4Dfastsim.clone(vertices="unsortedOfflinePrimaryVertices4Dfastsim:WithBS")

from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import tpClusterProducer
from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import quickTrackAssociatorByHits
from SimTracker.TrackAssociation.trackTimeValueMapProducer_cfi import trackTimeValueMapProducer
from CommonTools.RecoAlgos.tofPID_cfi import tofPID

tofPIDfor4DwithPID=tofPID.clone()
tofPID.vtxsSrc=cms.InputTag('unsortedOfflinePrimaryVertices4DwithPID')
