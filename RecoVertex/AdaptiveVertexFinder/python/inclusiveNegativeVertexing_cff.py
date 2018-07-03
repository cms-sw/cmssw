import FWCore.ParameterSet.Config as cms

from RecoVertex.AdaptiveVertexFinder.inclusiveCandidateVertexFinder_cfi import *
from RecoVertex.AdaptiveVertexFinder.candidateVertexMerger_cfi import *
from RecoVertex.AdaptiveVertexFinder.candidateVertexArbitrator_cfi import *


inclusiveCandidateNegativeVertexFinder = inclusiveCandidateVertexFinder.clone(
   vertexMinAngleCosine = -0.95, 
   clusterizer = dict(
        clusterMinAngleCosine = -0.5
        )
)

candidateNegativeVertexMerger = candidateVertexMerger.clone(
   secondaryVertices = "inclusiveCandidateNegativeVertexFinder"
)

candidateNegativeVertexArbitrator = candidateVertexArbitrator.clone(
   secondaryVertices = "candidateNegativeVertexMerger",
   dRCut = -0.4
)

inclusiveCandidateNegativeSecondaryVertices = candidateVertexMerger.clone(
   secondaryVertices = "candidateNegativeVertexArbitrator",
   maxFraction = 0.2,
   minSignificance = 10.
)


inclusiveCandidateNegativeVertexingTask = cms.Task(inclusiveCandidateNegativeVertexFinder,
                                           candidateNegativeVertexMerger,
                                           candidateNegativeVertexArbitrator,
                                           inclusiveCandidateNegativeSecondaryVertices)

inclusiveCandidateNegativeVertexing = cms.Sequence(inclusiveCandidateNegativeVertexingTask)


inclusiveCandidateNegativeVertexFinderCvsL = inclusiveCandidateVertexFinder.clone(
   vertexMinDLen2DSig = 1.25,
   vertexMinDLenSig = 0.25,
   vertexMinAngleCosine = -0.95,
   clusterizer = dict(
      clusterMinAngleCosine = -0.5
      )
)

candidateNegativeVertexMergerCvsL = candidateVertexMerger.clone(
   secondaryVertices = "inclusiveCandidateNegativeVertexFinderCvsL"
)

candidateNegativeVertexArbitratorCvsL = candidateVertexArbitrator.clone(
   secondaryVertices = "candidateNegativeVertexMergerCvsL",
   dRCut = -0.4
)

inclusiveCandidateNegativeSecondaryVerticesCvsL = candidateVertexMerger.clone(
   secondaryVertices = "candidateNegativeVertexArbitratorCvsL",
   maxFraction = 0.2,
   minSignificance = 10.
)


inclusiveCandidateNegativeVertexingCvsLTask = cms.Task(inclusiveCandidateNegativeVertexFinderCvsL,
                                               candidateNegativeVertexMergerCvsL,
                                               candidateNegativeVertexArbitratorCvsL,
                                               inclusiveCandidateNegativeSecondaryVerticesCvsL)
inclusiveCandidateNegativeVertexingCvsL = cms.Sequence(inclusiveCandidateNegativeVertexingCvsLTask)
