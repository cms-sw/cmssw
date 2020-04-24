import FWCore.ParameterSet.Config as cms

candidateVertexMerger = cms.EDProducer("CandidateVertexMerger",
       secondaryVertices = cms.InputTag("inclusiveCandidateVertexFinder"),
       maxFraction = cms.double(0.7),
       minSignificance = cms.double(2)
)


