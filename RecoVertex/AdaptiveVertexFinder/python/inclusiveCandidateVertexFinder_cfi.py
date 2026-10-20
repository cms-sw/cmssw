import FWCore.ParameterSet.Config as cms

from RecoVertex.AdaptiveVertexFinder.inclusiveCandidateVertexFinderDefault_cfi import inclusiveCandidateVertexFinderDefault

inclusiveCandidateVertexFinder  = inclusiveCandidateVertexFinderDefault.clone()

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
(pp_on_XeXe_2017 | pp_on_AA).toModify(inclusiveCandidateVertexFinder, minHits = 10, minPt = 1.0)
