import FWCore.ParameterSet.Config as cms

from TopQuarkAnalysis.TopSkimming.topMuSkimFilterOctoberX_cfi import *

#Define group sequence, using HLT/Reco quality cut.
#Removing it from the sequence for now
topMuHLTSeq = cms.Sequence(
	topMuHLT+topMuHLTPtFilter
)
