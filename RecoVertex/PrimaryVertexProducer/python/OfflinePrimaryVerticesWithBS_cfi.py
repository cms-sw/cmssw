import FWCore.ParameterSet.Config as cms

offlinePrimaryVerticesWithBS = cms.EDFilter("VertexSelector",
                                            src = cms.InputTag("offlinePrimaryVertices","WithBS"),
                                            cut = cms.string(""),
                                            filter = cms.bool(False)
)
                                              

# foo bar baz
# 1XcfVWjvx4sph
