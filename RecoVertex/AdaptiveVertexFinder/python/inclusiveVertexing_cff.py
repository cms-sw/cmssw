import FWCore.ParameterSet.Config as cms

from RecoVertex.AdaptiveVertexFinder.inclusiveVertexFinder_cfi import *
from RecoVertex.AdaptiveVertexFinder.vertexMerger_cfi import *
from RecoVertex.AdaptiveVertexFinder.trackVertexArbitrator_cfi import *

inclusiveSecondaryVertices = vertexMerger.clone(
    secondaryVertices = "trackVertexArbitrator",
    maxFraction = 0.2,
    minSignificance = 10.
)

inclusiveVertexingTask = cms.Task(inclusiveVertexFinder,
                                  vertexMerger,
                                  trackVertexArbitrator,
                                  inclusiveSecondaryVertices)
inclusiveVertexing = cms.Sequence(inclusiveVertexingTask)

from RecoVertex.AdaptiveVertexFinder.inclusiveCandidateVertexFinder_cfi import *
from RecoVertex.AdaptiveVertexFinder.candidateVertexMerger_cfi import *
from RecoVertex.AdaptiveVertexFinder.candidateVertexArbitrator_cfi import *

inclusiveCandidateSecondaryVertices = candidateVertexMerger.clone(
    secondaryVertices = "candidateVertexArbitrator",
    maxFraction = 0.2,
    minSignificance = 10.
)

inclusiveCandidateVertexingTask = cms.Task(inclusiveCandidateVertexFinder,
                                           candidateVertexMerger,
                                           candidateVertexArbitrator,
                                           inclusiveCandidateSecondaryVertices)
inclusiveCandidateVertexing = cms.Sequence(inclusiveCandidateVertexingTask)

#relaxed IVF reconstruction cuts for candidate-based ctagging
inclusiveCandidateVertexFinderCvsL = inclusiveCandidateVertexFinder.clone(
   vertexMinDLen2DSig = 1.25,
   vertexMinDLenSig = 0.25
)

candidateVertexMergerCvsL = candidateVertexMerger.clone(
   secondaryVertices = "inclusiveCandidateVertexFinderCvsL"
)

candidateVertexArbitratorCvsL = candidateVertexArbitrator.clone(
   secondaryVertices = cms.InputTag("candidateVertexMergerCvsL")
)

inclusiveCandidateSecondaryVerticesCvsL = candidateVertexMerger.clone(
   secondaryVertices = "candidateVertexArbitratorCvsL",
   maxFraction = 0.2,
   minSignificance = 10.
)

inclusiveCandidateVertexingCvsLTask = cms.Task(inclusiveCandidateVertexFinderCvsL,
                                               candidateVertexMergerCvsL,
                                               candidateVertexArbitratorCvsL,
                                               inclusiveCandidateSecondaryVerticesCvsL)
inclusiveCandidateVertexingCvsL = cms.Sequence(inclusiveCandidateVertexingCvsLTask)

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
pp_on_XeXe_2017.toModify(inclusiveVertexFinder, minHits = 10, minPt = 1.0)
pp_on_XeXe_2017.toModify(inclusiveCandidateVertexFinder, minHits = 10, minPt = 1.0)
pp_on_XeXe_2017.toModify(inclusiveCandidateVertexFinderCvsL, minHits = 10, minPt = 1.0)
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
pp_on_AA_2018.toModify(inclusiveVertexFinder, minHits = 999, minPt = 999.0)
pp_on_AA_2018.toModify(inclusiveCandidateVertexFinder, minHits = 999, minPt = 999.0)
pp_on_AA_2018.toModify(inclusiveCandidateVertexFinderCvsL, minHits = 999, minPt = 999.0)


