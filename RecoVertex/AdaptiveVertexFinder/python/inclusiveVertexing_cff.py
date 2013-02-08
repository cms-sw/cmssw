import FWCore.ParameterSet.Config as cms

from RecoVertex.AdaptiveVertexFinder.inclusiveVertexFinder_cfi import *
from RecoVertex.AdaptiveVertexFinder.vertexMerger_cfi import *
from RecoVertex.AdaptiveVertexFinder.trackVertexArbitrator_cfi import *

inclusiveVertices = trackVertexArbitrator.clone()

inclusiveVertexing = cms.Sequence(inclusiveVertexFinder*vertexMerger*inclusiveVertices)

