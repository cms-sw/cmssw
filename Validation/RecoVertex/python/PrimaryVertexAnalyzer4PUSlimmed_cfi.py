import FWCore.ParameterSet.Config as cms

selectedOfflinePrimaryVertices = cms.EDFilter("VertexSelector",
                                               src = cms.InputTag('offlinePrimaryVertices'),
                                               cut = cms.string("isValid & ndof > 4 & tracksSize > 0 & abs(z) <= 24 & abs(position.Rho) <= 2."),
                                               filter = cms.bool(False)
)

selectedOfflinePrimaryVerticesWithBS = selectedOfflinePrimaryVertices.clone(
  src = 'offlinePrimaryVerticesWithBS'
)
selectedPixelVertices = selectedOfflinePrimaryVertices.clone(
  src = 'pixelVertices'
)
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
vertexAnalysis = DQMEDAnalyzer('PrimaryVertexAnalyzer4PUSlimmed',
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
                                                                      "selectedOfflinePrimaryVerticesWithBS"
                                                                      ),
                               nPUbins = cms.uint32(130)
)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify( vertexAnalysis,
                         nPUbins = 250 )

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(vertexAnalysis,
    trackingParticleCollection = "mixData:MergedTrackTruth",
    trackingVertexCollection = "mixData:MergedTrackTruth",
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
    vertexAssociator = "PixelVertexAssociatorByPositionAndTracks",
    vertexRecoCollections = [
        "pixelVertices",
        "selectedPixelVertices"
    ]
)
pixelVertexAnalysisPixelTrackingOnly = pixelVertexAnalysisTrackingOnly.clone(
    do_generic_sim_plots = True,
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

_vertexAnalysisSequenceTrackingOnly_trackingLowPU = vertexAnalysisSequenceTrackingOnly.copy()
_vertexAnalysisSequenceTrackingOnly_trackingLowPU += (
    selectedPixelVertices
    + pixelVertexAnalysisTrackingOnly
)
trackingLowPU.toReplaceWith(vertexAnalysisSequenceTrackingOnly, _vertexAnalysisSequenceTrackingOnly_trackingLowPU)

vertexAnalysisSequencePixelTrackingOnly = cms.Sequence(
    selectedPixelVertices
    + pixelVertexAnalysisPixelTrackingOnly
)



from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
_vertexRecoCollectionsTiming = cms.VInputTag("offlinePrimaryVertices",
                                             "offlinePrimaryVerticesWithBS",
                                             "selectedOfflinePrimaryVertices",
                                             "selectedOfflinePrimaryVerticesWithBS",
                                             "offlinePrimaryVertices4D",
                                             "selectedOfflinePrimaryVertices4D",
                                             )
selectedOfflinePrimaryVertices4D = selectedOfflinePrimaryVertices.clone(src = cms.InputTag("offlinePrimaryVertices4D"))

_vertexAnalysisSelectionTiming = vertexAnalysisSelection.copy()
_vertexAnalysisSelectionTiming += selectedOfflinePrimaryVertices4D

phase2_timing_layer.toModify( vertexAnalysis, 
                              vertexRecoCollections = _vertexRecoCollectionsTiming
                              )
phase2_timing_layer.toReplaceWith( vertexAnalysisSelection,
                                   _vertexAnalysisSelectionTiming )
