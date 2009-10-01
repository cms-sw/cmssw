import FWCore.ParameterSet.Config as cms

from TopQuarkAnalysis.TopSkimming.topMuSkimFilterOctoberX_cfi import *

#Define group sequence, using HLT/Reco quality cut. 
topMuHLTSeq = cms.Sequence(
	topMuHLT+topMuHLTPtFilter
#    	topMuHLTPtFilter
#        topMuHLT
)
