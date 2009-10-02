import FWCore.ParameterSet.Config as cms

from HLTrigger.HLTfilters.hltHighLevel_cfi import *
topMuHLT = hltHighLevel.clone()
topMuHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29")
#Define the HLT path to be used.
topMuHLT.HLTPaths =cms.vstring('HLT_Mu9')


from HLTrigger.HLTfilters.hltSummaryFilter_cfi import *
#Define the second filter using the pt of the L3MuonCollection:
topMuHLTPtFilter=hltSummaryFilter.clone()
topMuHLTPtFilter.summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29")
topMuHLTPtFilter.member  = cms.InputTag("hltSingleMu9L3Filtered9","","HLT8E29") 
topMuHLTPtFilter.cut = cms.string("pt>18")
