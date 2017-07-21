import FWCore.ParameterSet.Config as cms

selectedOfflinePrimaryVertices = cms.EDFilter("VertexSelector",
                                               src = cms.InputTag('offlinePrimaryVertices'),
                                               cut = cms.string("isValid & ndof > 4 & tracksSize > 0 & abs(z) <= 24 & abs(position.Rho) <= 2."),
                                               filter = cms.bool(False)
)

selectedOfflinePrimaryVerticesWithBS = selectedOfflinePrimaryVertices.clone()
selectedOfflinePrimaryVerticesWithBS.src = cms.InputTag('offlinePrimaryVerticesWithBS')

selectedPixelVertices = selectedOfflinePrimaryVertices.clone()
selectedPixelVertices.src = cms.InputTag('pixelVertices')

vertexAnalysis = cms.EDAnalyzer("PrimaryVertexAnalyzer4PUSlimmed",
                                use_only_charged_tracks = cms.untracked.bool(True),
                                do_generic_sim_plots = cms.untracked.bool(True),
                                verbose = cms.untracked.bool(False),
                                root_folder = cms.untracked.string("Vertexing/PrimaryVertexV"),
                                trackingParticleCollection = cms.untracked.InputTag("mix", "MergedTrackTruth"),
                                trackingVertexCollection = cms.untracked.InputTag("mix", "MergedTrackTruth"),
                                trackAssociatorMap = cms.untracked.InputTag("trackingParticleRecoTrackAsssociation"),
                                vertexAssociator = cms.untracked.InputTag("VertexAssociatorByPositionAndTracks"),
                                vertexRecoCollections = cms.VInputTag("offlinePrimaryVertices",
                                                                      "offlinePrimaryVerticesWithBS",
                                                                      "selectedOfflinePrimaryVertices",
                                                                      "selectedOfflinePrimaryVerticesWithBS",
                                ),
)

vertexAnalysisTrackingOnly = vertexAnalysis.clone(
    vertexRecoCollections = vertexAnalysis.vertexRecoCollections.value() + [
        "firstStepPrimaryVerticesPreSplitting",
        "firstStepPrimaryVertices"
    ]
)
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toModify(vertexAnalysisTrackingOnly, vertexRecoCollections = vertexAnalysis.vertexRecoCollections.value())
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(vertexAnalysisTrackingOnly,
    vertexRecoCollections = vertexAnalysis.vertexRecoCollections.value() + [
        "firstStepPrimaryVertices"
    ]
)

pixelVertexAnalysisTrackingOnly = vertexAnalysis.clone(
    do_generic_sim_plots = False,
    trackAssociatorMap = "trackingParticlePixelTrackAsssociation",
    vertexRecoCollections = [
        "pixelVertices",
        "selectedPixelVertices"
    ]
)

##########

vertexAnalysisSelection = cms.Sequence(
    cms.ignore(selectedOfflinePrimaryVertices)
    + cms.ignore(selectedOfflinePrimaryVerticesWithBS)
)

##########

vertexAnalysisSequence = cms.Sequence(
    vertexAnalysisSelection
    + vertexAnalysis
)

vertexAnalysisSequenceTrackingOnly = cms.Sequence(
    vertexAnalysisSelection
    + vertexAnalysisTrackingOnly
)

from SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi import trackingParticleRecoTrackAsssociation as _trackingParticleRecoTrackAsssociation
trackingParticlePixelTrackAsssociation = _trackingParticleRecoTrackAsssociation.clone(
    label_tr = "pixelTracks"
)

_vertexAnalysisSequenceTrackingOnly_trackingLowPU = vertexAnalysisSequenceTrackingOnly.copy()
_vertexAnalysisSequenceTrackingOnly_trackingLowPU += (
    trackingParticlePixelTrackAsssociation
    + selectedPixelVertices
    + pixelVertexAnalysisTrackingOnly
)
trackingLowPU.toReplaceWith(vertexAnalysisSequenceTrackingOnly, _vertexAnalysisSequenceTrackingOnly_trackingLowPU)
