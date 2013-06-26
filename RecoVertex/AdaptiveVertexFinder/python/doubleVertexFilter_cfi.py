import FWCore.ParameterSet.Config as cms

doubleVertexFilter = cms.EDProducer("DoubleVertexFilter",
       primaryVertices = cms.InputTag("offlinePrimaryVertices"),
       secondaryVertices = cms.InputTag("vertexFinder"),
       maxFraction = cms.double(0.7)
)


