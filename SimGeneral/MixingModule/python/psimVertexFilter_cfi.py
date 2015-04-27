import FWCore.ParameterSet.Config as cms

psimVertexFilter = cms.EDFilter("PSimVertexFilter",
                                simVtxTag = cms.InputTag("mix", "g4SimHits", "DIGI2RAW")
                                )

