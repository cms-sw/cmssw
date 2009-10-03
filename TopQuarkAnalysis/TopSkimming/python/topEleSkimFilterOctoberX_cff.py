import FWCore.ParameterSet.Config as cms

from TopQuarkAnalysis.TopSkimming.topEleSkimFilterOctoberX_cfi import *

#Have to have 3 seperate so we can do an OR later 
topHLT_Ele15_LW_L1R_Seq      = cms.Sequence( topEleHLT + topHLT_Ele15_LW_L1R_PtFilter)
topHLT_Ele15_SC10_LW_L1R_Seq = cms.Sequence( topEleHLT + topHLT_Ele15_SC10_LW_L1R_PtFilter)
topHLT_Ele20_LW_L1R_Seq      = cms.Sequence( topEleHLT + topHLT_Ele20_LW_L1R_PtFilter)


