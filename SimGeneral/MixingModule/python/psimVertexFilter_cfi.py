import FWCore.ParameterSet.Config as cms

psimVertexFilter = cms.EDFilter("PSimVertexFilter",
                                simVtxTag = cms.InputTag("cfWriter", "g4SimHits", "DIGI2RAW")
                                )

