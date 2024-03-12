import FWCore.ParameterSet.Config as cms

vertexConstraint = cms.EDProducer(
    "VertexConstraintProducer",
    srcTrk = cms.InputTag("AlignmentTrackSelector"),
    srcVtx = cms.InputTag("offlinePrimaryVertices")
)
# foo bar baz
# 2Xzt2GnGtcL0s
# U1ajVJFn19BRk
