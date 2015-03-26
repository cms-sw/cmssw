import FWCore.ParameterSet.Config as cms

from RecoVertex.AdaptiveVertexFinder.inclusiveVertexFinder_cfi import *
from RecoVertex.AdaptiveVertexFinder.vertexMerger_cfi import *
from RecoVertex.AdaptiveVertexFinder.trackVertexArbitrator_cfi import *

inclusiveSecondaryVertices = vertexMerger.clone()
inclusiveSecondaryVertices.secondaryVertices = cms.InputTag("trackVertexArbitrator")
inclusiveSecondaryVertices.maxFraction = 0.2
inclusiveSecondaryVertices.minSignificance = 10.

inclusiveVertexing = cms.Sequence(inclusiveVertexFinder*vertexMerger*trackVertexArbitrator*inclusiveSecondaryVertices)

from RecoVertex.AdaptiveVertexFinder.inclusiveCandidateVertexFinder_cfi import *
from RecoVertex.AdaptiveVertexFinder.candidateVertexMerger_cfi import *
from RecoVertex.AdaptiveVertexFinder.candidateVertexArbitrator_cfi import *

inclusiveCandidateSecondaryVertices = candidateVertexMerger.clone()
inclusiveCandidateSecondaryVertices.secondaryVertices = cms.InputTag("candidateVertexArbitrator")
inclusiveCandidateSecondaryVertices.maxFraction = 0.2
inclusiveCandidateSecondaryVertices.minSignificance = 10.

inclusiveCandidateVertexing = cms.Sequence(inclusiveCandidateVertexFinder*candidateVertexMerger*candidateVertexArbitrator*inclusiveCandidateSecondaryVertices)


#relaxed IVF reconstruction cuts for candidate-based ctagging
inclusiveCandidateVertexFinderCtagL = inclusiveCandidateVertexFinder.clone()
inclusiveCandidateVertexFinderCtagL.vertexMinDLen2DSig = 1.25 
inclusiveCandidateVertexFinderCtagL.vertexMinDLenSig = 0.25
inclusiveCandidateVertexFinderCtagL.clusterizer.seedMin3DIPSignificance = 1.0
#inclusiveCandidateVertexFinderCtagL.clusterizer.seedMin3DIPValue = 0.005
inclusiveCandidateVertexFinderCtagL.clusterizer.distanceRatio = 10

candidateVertexMergerCtagL = candidateVertexMerger.clone()
candidateVertexMergerCtagL.secondaryVertices = cms.InputTag("inclusiveCandidateVertexFinderCtagL")

candidateVertexArbitratorCtagL = candidateVertexArbitrator.clone()
candidateVertexArbitratorCtagL.secondaryVertices = cms.InputTag("candidateVertexMergerCtagL")

inclusiveCandidateSecondaryVerticesCtagL = candidateVertexMerger.clone()
inclusiveCandidateSecondaryVerticesCtagL.secondaryVertices = cms.InputTag("candidateVertexArbitratorCtagL")
inclusiveCandidateSecondaryVerticesCtagL.maxFraction = 0.2
inclusiveCandidateSecondaryVerticesCtagL.minSignificance = 10.

inclusiveCandidateVertexingCtagL = cms.Sequence(inclusiveCandidateVertexFinderCtagL*candidateVertexMergerCtagL*candidateVertexArbitratorCtagL*inclusiveCandidateSecondaryVerticesCtagL)


