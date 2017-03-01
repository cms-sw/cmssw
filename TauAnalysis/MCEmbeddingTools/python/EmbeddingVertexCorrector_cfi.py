import FWCore.ParameterSet.Config as cms


VtxCorrectedToInput = cms.EDProducer("EmbeddingVertexCorrector",
	src = cms.InputTag("generator","unsmeared")
)