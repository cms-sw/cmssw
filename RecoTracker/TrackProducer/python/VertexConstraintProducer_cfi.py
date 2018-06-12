import FWCore.ParameterSet.Config as cms

vertexConstraint = cms.EDProducer(
    "VertexConstraintProducer",
    srcTrk = cms.InputTag("AlignmentTrackSelector"),
    srcVtx = cms.InputTag("offlinePrimaryVertices")
)
