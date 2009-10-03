import FWCore.ParameterSet.Config as cms

from HLTrigger.HLTfilters.hltHighLevel_cfi import *
from HLTrigger.HLTfilters.hltSummaryFilter_cfi import *

#don't actually need these for the october exercise
topEleHLT = hltHighLevel.clone()
topEleHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29")
#Define the HLT path to be used.
topEleHLT.HLTPaths =cms.vstring('HLT_Ele15_LW_L1R','HLT_Ele15_SC10_LW_L1R','HLT_Ele20_LW_L1R')


#Filter for HLT_Ele15_LW_L1R is hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter
topHLT_Ele15_LW_L1R_PtFilter=hltSummaryFilter.clone()
topHLT_Ele15_LW_L1R_PtFilter.summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29")
topHLT_Ele15_LW_L1R_PtFilter.member  = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter","","HLT8E29") 
topHLT_Ele15_LW_L1R_PtFilter.cut = cms.string("pt>18")

#Filter for HLT_Ele15_SC10_LW_L1R is hltL1NonIsoHLTNonIsoSingleElectronLWEt15ESDoubleSC10
from HLTrigger.HLTfilters.hltSummaryFilter_cfi import *
topHLT_Ele15_SC10_LW_L1R_PtFilter=hltSummaryFilter.clone()
topHLT_Ele15_SC10_LW_L1R_PtFilter.summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29")
topHLT_Ele15_SC10_LW_L1R_PtFilter.member  = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15ESDoubleSC10","","HLT8E29") 
topHLT_Ele15_SC10_LW_L1R_PtFilter.cut = cms.string("pt>18")

#Filter for HLT_Ele20_LW_L1R is hltL1NonIsoHLTNonIsoSingleElectronLWEt15ESDoubleSC10
from HLTrigger.HLTfilters.hltSummaryFilter_cfi import *
topHLT_Ele20_LW_L1R_PtFilter=hltSummaryFilter.clone()
topHLT_Ele20_LW_L1R_PtFilter.summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29")
topHLT_Ele20_LW_L1R_PtFilter.member  = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15EtFilterESet20","","HLT8E29") 
topHLT_Ele20_LW_L1R_PtFilter.cut = cms.string("pt>18")
