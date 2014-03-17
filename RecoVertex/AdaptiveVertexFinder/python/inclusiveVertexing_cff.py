import FWCore.ParameterSet.Config as cms

from RecoVertex.AdaptiveVertexFinder.inclusiveVertexFinder_cfi import *
from RecoVertex.AdaptiveVertexFinder.vertexMerger_cfi import *
from RecoVertex.AdaptiveVertexFinder.trackVertexArbitrator_cfi import *

inclusiveVertices = vertexMerger.clone()
inclusiveVertices.secondaryVertices = cms.InputTag("trackVertexArbitrator")
inclusiveVertices.maxFraction = 0.2
inclusiveVertices.minSignificance = 10.

inclusiveVertexing = cms.Sequence(inclusiveVertexFinder*vertexMerger*trackVertexArbitrator*inclusiveVertices)

