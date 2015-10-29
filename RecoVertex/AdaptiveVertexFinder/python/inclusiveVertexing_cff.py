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
inclusiveCandidateVertexFinderCvsL = inclusiveCandidateVertexFinder.clone(
   vertexMinDLen2DSig = cms.double(1.25),
   vertexMinDLenSig = cms.double(0.25)
)

candidateVertexMergerCvsL = candidateVertexMerger.clone(
   secondaryVertices = cms.InputTag("inclusiveCandidateVertexFinderCvsL")
)

candidateVertexArbitratorCvsL = candidateVertexArbitrator.clone(
   secondaryVertices = cms.InputTag("candidateVertexMergerCvsL")
)

inclusiveCandidateSecondaryVerticesCvsL = candidateVertexMerger.clone(
   secondaryVertices = cms.InputTag("candidateVertexArbitratorCvsL"),
   maxFraction = cms.double(0.2),
   minSignificance = cms.double(10.)
)

inclusiveCandidateVertexingCvsL = cms.Sequence(inclusiveCandidateVertexFinderCvsL*candidateVertexMergerCvsL*candidateVertexArbitratorCvsL*inclusiveCandidateSecondaryVerticesCvsL)


