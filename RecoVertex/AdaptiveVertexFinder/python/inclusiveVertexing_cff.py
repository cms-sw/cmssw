import FWCore.ParameterSet.Config as cms

from RecoVertex.AdaptiveVertexFinder.inclusiveVertexFinder_cfi import *
from RecoVertex.AdaptiveVertexFinder.vertexMerger_cfi import *
from RecoVertex.AdaptiveVertexFinder.trackVertexArbitrator_cfi import *

inclusiveMergedVertices = vertexMerger.clone()
inclusiveMergedVertices.secondaryVertices = cms.InputTag("trackVertexArbitrator")
inclusiveMergedVertices.maxFraction = 0.2
inclusiveMergedVertices.minSignificance = 10.

inclusiveVertexing = cms.Sequence(inclusiveVertexFinder*vertexMerger*trackVertexArbitrator*inclusiveMergedVertices)

