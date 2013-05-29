import FWCore.ParameterSet.Config as cms

offlinePrimaryVerticesWithBS = cms.EDFilter("VertexSelector",
                                            src = cms.InputTag("offlinePrimaryVertices","WithBS"),
                                            cut = cms.string(""),
                                            filter = cms.bool(False)
)
                                              

