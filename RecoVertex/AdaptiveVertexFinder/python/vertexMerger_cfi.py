import FWCore.ParameterSet.Config as cms

vertexMerger = cms.EDProducer("VertexMerger",
       secondaryVertices = cms.InputTag("vertexFinder"),
       maxFraction = cms.double(0.7)
)


