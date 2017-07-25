import FWCore.ParameterSet.Config as cms

VertexConstraint = cms.EDProducer(
    "VertexConstraintProducer",
    srcTrk = cms.InputTag("AlignmentTrackSelector"),
    srcVtx = cms.InputTag("offlinePrimaryVertices")
)
