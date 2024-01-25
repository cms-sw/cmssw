import FWCore.ParameterSet.Config as cms
from RecoVertex.Configuration.RecoVertex_cff import unsortedOfflinePrimaryVertices, trackWithVertexRefSelector, trackRefsForJets, sortedPrimaryVertices, offlinePrimaryVertices, offlinePrimaryVerticesWithBS,vertexrecoTask

unsortedOfflinePrimaryVertices4D = unsortedOfflinePrimaryVertices.clone(
    TkClusParameters = dict(
        algorithm = "DA2D_vect", 
        TkDAClusParameters = dict(
            Tmin = 4.0, 
            Tpurge = 4.0, 
            Tstop = 2.0
        ),
    ),
    TrackTimesLabel = cms.InputTag("trackTimeValueMapProducer","generalTracksConfigurableFlatResolutionModel"),
    TrackTimeResosLabel = cms.InputTag("trackTimeValueMapProducer","generalTracksConfigurableFlatResolutionModelResolution"),
    vertexCollections = {0: dict(vertexTimeParameters = cms.PSet( algorithm = cms.string('legacy4D'))),
                         1: dict(vertexTimeParameters = cms.PSet( algorithm = cms.string('legacy4D')))}
)
trackWithVertexRefSelectorBeforeSorting4D = trackWithVertexRefSelector.clone(
    vertexTag = "unsortedOfflinePrimaryVertices4D",
    ptMax = 9e99,
    ptErrorCut = 9e99
)
trackRefsForJetsBeforeSorting4D = trackRefsForJets.clone(
    src = "trackWithVertexRefSelectorBeforeSorting4D"
)
offlinePrimaryVertices4D = sortedPrimaryVertices.clone(
    vertices = "unsortedOfflinePrimaryVertices4D",
    particles = "trackRefsForJetsBeforeSorting4D",
    trackTimeTag = "trackTimeValueMapProducer:generalTracksConfigurableFlatResolutionModel",
    trackTimeResoTag = "trackTimeValueMapProducer:generalTracksConfigurableFlatResolutionModelResolution",
    assignment = dict(useTiming = True)
)
offlinePrimaryVertices4DWithBS = offlinePrimaryVertices4D.clone(
    vertices = "unsortedOfflinePrimaryVertices4D:WithBS"
)

unsortedOfflinePrimaryVertices4DwithPID = unsortedOfflinePrimaryVertices4D.clone(
    TrackTimesLabel = "tofPID4DnoPID:t0safe",
    TrackTimeResosLabel = "tofPID4DnoPID:sigmat0safe"
)
trackWithVertexRefSelectorBeforeSorting4DwithPID = trackWithVertexRefSelector.clone(
    vertexTag = "unsortedOfflinePrimaryVertices4DwithPID",
    ptMax = 9e99,
    ptErrorCut = 9e99
)
trackRefsForJetsBeforeSorting4DwithPID = trackRefsForJets.clone(
    src = "trackWithVertexRefSelectorBeforeSorting4DwithPID"
)
offlinePrimaryVertices4DwithPID=offlinePrimaryVertices4D.clone(
    vertices = "unsortedOfflinePrimaryVertices4DwithPID",
    particles = "trackRefsForJetsBeforeSorting4DwithPID",
    trackTimeTag = "tofPID4DnoPID:t0safe",
    trackTimeResoTag = "tofPID4DnoPID:sigmat0safe"
)
offlinePrimaryVertices4DwithPIDWithBS = offlinePrimaryVertices4DwithPID.clone(
    vertices = "unsortedOfflinePrimaryVertices4DwithPID:WithBS"
)

from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import tpClusterProducer
from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import quickTrackAssociatorByHits
from SimTracker.TrackAssociation.trackTimeValueMapProducer_cfi import trackTimeValueMapProducer
from RecoMTD.TimingIDTools.tofPIDProducer_cfi import tofPIDProducer

tofPID4DnoPID=tofPIDProducer.clone(vtxsSrc='unsortedOfflinePrimaryVertices')
tofPID=tofPIDProducer.clone()
tofPID3D=tofPIDProducer.clone(vtxsSrc='unsortedOfflinePrimaryVertices')

from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
phase2_timing_layer.toModify(tofPID, vtxsSrc='unsortedOfflinePrimaryVertices4D', vertexReassignment=False)
phase2_timing_layer.toModify(tofPID3D, vertexReassignment=False)
phase2_timing_layer.toModify(unsortedOfflinePrimaryVertices, 
    vertexCollections = {0: dict(vertexTimeParameters = cms.PSet( algorithm = cms.string('fromTracksPID'))),
                         1: dict(vertexTimeParameters = cms.PSet( algorithm = cms.string('fromTracksPID')))}
)

