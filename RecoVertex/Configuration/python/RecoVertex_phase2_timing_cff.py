import FWCore.ParameterSet.Config as cms
from RecoVertex.Configuration.RecoVertex_cff import unsortedOfflinePrimaryVertices, trackWithVertexRefSelector, trackRefsForJets, sortedPrimaryVertices, offlinePrimaryVertices, offlinePrimaryVerticesWithBS,vertexrecoTask

unsortedOfflinePrimaryVertices4D = unsortedOfflinePrimaryVertices.clone(
    TkClusParameters = cms.PSet(algorithm = cms.string("DA2D_vect"),
        TkDAClusParameters = cms.PSet(
            Tmin = cms.double(4.0),
            Tpurge = cms.double(4.0),
            Tstop = cms.double(2.0),
        )
    ),
    TrackTimesLabel = cms.InputTag("trackTimeValueMapProducer","generalTracksConfigurableFlatResolutionModel"),
    TrackTimeResosLabel = cms.InputTag("trackTimeValueMapProducer","generalTracksConfigurableFlatResolutionModelResolution"),
    trackMTDTimeQualityVMapTag = cms.InputTag("mtdTrackQualityMVA:mtdQualMVA"),
    useMVACut = cms.bool(False),
    minTrackTimeQuality = cms.double(0.8),
    vertexCollections = {0: dict(vertexTimeParameters = cms.PSet( algorithm = cms.string('fromTracksPID'))),
                         1: dict(vertexTimeParameters = cms.PSet( algorithm = cms.string('fromTracksPID')))}
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

trackFeatureProducer = cms.EDProducer("vertexgnn::TrackFeatureProducer",
    TkFilterParameters = unsortedOfflinePrimaryVertices.TkFilterParameters.clone()
)
gnnVertexProducer = cms.EDProducer("vertexgnn::GNNVertexProducerAlpaka@alpaka",
    trackFeatures = cms.InputTag("trackFeatureProducer"),
    model = cms.FileInPath("RecoVertex/PrimaryVertexProducer/data/vertexSlotGNN.pt"),
)
unsortedOfflinePrimaryVerticesGNN = unsortedOfflinePrimaryVertices4D.clone(
    TkClusParameters = cms.PSet(algorithm = cms.string("GNN2D_alpaka"),
        TkDAClusParameters = cms.PSet(
            existenceThreshold = cms.double(0.01),
            trackAssignmentThreshold = cms.double(0.5),
            gnnOutput = cms.InputTag("gnnVertexProducer"),
        )
    ),
    TrackTimesLabel = "tofPID4DnoPID:t0safe",
    TrackTimeResosLabel = "tofPID4DnoPID:sigmat0safe",
    vertexCollections = {0: dict(useClusterWeights = cms.bool(False)),
                         1: dict(useClusterWeights = cms.bool(False))}
)
trackWithVertexRefSelectorBeforeSortingGNN = trackWithVertexRefSelector.clone(
    vertexTag = "unsortedOfflinePrimaryVerticesGNN",
    ptMax = 9e99,
    ptErrorCut = 9e99
)
trackRefsForJetsBeforeSortingGNN = trackRefsForJets.clone(
    src = "trackWithVertexRefSelectorBeforeSortingGNN"
)
offlinePrimaryVerticesGNN = sortedPrimaryVertices.clone(
    vertices = "unsortedOfflinePrimaryVerticesGNN",
    particles = "trackRefsForJetsBeforeSortingGNN",
    trackTimeTag = "tofPID4DnoPID:t0safe",
    trackTimeResoTag = "tofPID4DnoPID:sigmat0safe",
    assignment = dict(useTiming = True)
)
tofPIDGNN = tofPIDProducer.clone(vtxsSrc = 'unsortedOfflinePrimaryVerticesGNN')
phase2_timing_layer.toModify(tofPIDGNN, vertexReassignment = False)
