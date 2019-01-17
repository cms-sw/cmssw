import FWCore.ParameterSet.Config as cms
from RecoVertex.Configuration.RecoVertex_cff import unsortedOfflinePrimaryVertices, trackWithVertexRefSelector, trackRefsForJets, sortedPrimaryVertices, offlinePrimaryVertices, offlinePrimaryVerticesWithBS,vertexrecoTask

from RecoVertex.PrimaryVertexProducer.TkClusParameters_cff import DA2D_vectParameters
unsortedOfflinePrimaryVertices4DnoPID = unsortedOfflinePrimaryVertices.clone(TkClusParameters = DA2D_vectParameters,
                                                                         TrackTimesLabel = cms.InputTag("trackExtenderWithMTD:generalTrackt0"),
                                                                         TrackTimeResosLabel = cms.InputTag("trackExtenderWithMTD:generalTracksigmat0"),
                                                                         )
trackWithVertexRefSelectorBeforeSorting4DnoPID = trackWithVertexRefSelector.clone(vertexTag="unsortedOfflinePrimaryVertices4DnoPID",
                                                                                  ptMax=9e99,
                                                                                  ptErrorCut=9e99)
trackRefsForJetsBeforeSorting4DnoPID = trackRefsForJets.clone(src="trackWithVertexRefSelectorBeforeSorting4DnoPID")
offlinePrimaryVertices4DnoPID=sortedPrimaryVertices.clone(vertices="unsortedOfflinePrimaryVertices4DnoPID",
                                                          particles="trackRefsForJetsBeforeSorting4DnoPID",
                                                          trackTimeTag=cms.InputTag("trackExtenderWithMTD","generalTrackt0"),
                                                          trackTimeResoTag=cms.InputTag("trackExtenderWithMTD","generalTracksigmat0"),
                                                          assignment=dict(useTiming=True))
offlinePrimaryVertices4DnoPIDWithBS=sortedPrimaryVertices.clone(vertices="unsortedOfflinePrimaryVertices4DnoPID:WithBS",
                                                                particles="trackRefsForJetsBeforeSorting4DnoPID",
                                                                trackTimeTag=cms.InputTag("trackExtenderWithMTD","generalTrackt0"),
                                                                trackTimeResoTag=cms.InputTag("trackExtenderWithMTD","generalTracksigmat0"),
                                                                assignment=dict(useTiming=True))

unsortedOfflinePrimaryVertices4D = unsortedOfflinePrimaryVertices4DnoPID.clone(TrackTimesLabel = "tofPID:t0safe",
                                                                                 TrackTimeResosLabel = "tofPID:sigmat0safe",
                                                                                 )
trackWithVertexRefSelectorBeforeSorting4D = trackWithVertexRefSelector.clone(vertexTag="unsortedOfflinePrimaryVertices4D",
                                                                             ptMax=9e99,
                                                                             ptErrorCut=9e99)
trackRefsForJetsBeforeSorting4D = trackRefsForJets.clone(src="trackWithVertexRefSelectorBeforeSorting4D")
offlinePrimaryVertices4D=sortedPrimaryVertices.clone(vertices="unsortedOfflinePrimaryVertices4D",
                                                     particles="trackRefsForJetsBeforeSorting4D",
                                                     trackTimeTag=cms.InputTag("tofPID","t0safe"),
                                                     trackTimeResoTag=cms.InputTag("tofPID","sigmat0safe"),
                                                     assignment=dict(useTiming=True))
offlinePrimaryVertices4DWithBS=sortedPrimaryVertices.clone(vertices="unsortedOfflinePrimaryVertices4D:WithBS",
                                                           particles="trackRefsForJetsBeforeSorting4D",
                                                           trackTimeTag=cms.InputTag("tofPID","t0safe"),
                                                           trackTimeResoTag=cms.InputTag("tofPID","sigmat0safe"),
                                                           assignment=dict(useTiming=True))

unsortedOfflinePrimaryVertices4Dfastsim = unsortedOfflinePrimaryVertices4DnoPID.clone(TrackTimesLabel = "trackTimeValueMapProducer:generalTracksConfigurableFlatResolutionModel",
                                                                                 TrackTimeResosLabel = "trackTimeValueMapProducer:generalTracksConfigurableFlatResolutionModelResolution",
                                                                                 )
trackWithVertexRefSelectorBeforeSorting4Dfastsim = trackWithVertexRefSelector.clone(vertexTag="unsortedOfflinePrimaryVertices4Dfastsim",
                                                                                    ptMax=9e99,
                                                                                    ptErrorCut=9e99)
trackRefsForJetsBeforeSorting4Dfastsim = trackRefsForJets.clone(src="trackWithVertexRefSelectorBeforeSorting4Dfastsim")
offlinePrimaryVertices4Dfastsim=sortedPrimaryVertices.clone(vertices="unsortedOfflinePrimaryVertices4Dfastsim",
                                                            particles="trackRefsForJetsBeforeSorting4Dfastsim",
                                                            trackTimeTag=cms.InputTag("trackTimeValueMapProducer","generalTracksConfigurableFlatResolutionModel"),
                                                            trackTimeResoTag=cms.InputTag("trackTimeValueMapProducer","generalTracksConfigurableFlatResolutionModelResolution"),
                                                            assignment=dict(useTiming=True))
offlinePrimaryVertices4DfastsimWithBS=sortedPrimaryVertices.clone(vertices="unsortedOfflinePrimaryVertices4Dfastsim:WithBS",
                                                                  particles="trackRefsForJetsBeforeSorting4Dfastsim",
                                                                  trackTimeTag=cms.InputTag("trackTimeValueMapProducer","generalTracksConfigurableFlatResolutionModel"),
                                                                  trackTimeResoTag=cms.InputTag("trackTimeValueMapProducer","generalTracksConfigurableFlatResolutionModelResolution"),
                                                                  assignment=dict(useTiming=True))



unsortedOfflinePrimaryVertices3D = unsortedOfflinePrimaryVertices.clone()
trackWithVertexRefSelectorBeforeSorting3D = trackWithVertexRefSelector.clone(vertexTag="unsortedOfflinePrimaryVertices3D",
                                                                             ptMax=9e99,
                                                                             ptErrorCut=9e99)
trackRefsForJetsBeforeSorting3D = trackRefsForJets.clone(src="trackWithVertexRefSelectorBeforeSorting3D")
offlinePrimaryVertices3D = sortedPrimaryVertices.clone(vertices="unsortedOfflinePrimaryVertices3D",
                                                       particles="trackRefsForJetsBeforeSorting3D")
offlinePrimaryVertices3DWithBS = offlinePrimaryVerticesWithBS.clone(vertices="unsortedOfflinePrimaryVertices3D:WithBS",
                                                                    particles="trackRefsForJetsBeforeSorting3D")

from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import tpClusterProducer
from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import quickTrackAssociatorByHits
from SimTracker.TrackAssociation.trackTimeValueMapProducer_cfi import trackTimeValueMapProducer
from CommonTools.RecoAlgos.TOFPIDProducer_cfi import tofPID
