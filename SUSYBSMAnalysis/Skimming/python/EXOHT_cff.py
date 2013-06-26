

import FWCore.ParameterSet.Config as cms

from HLTrigger.HLTfilters.hltHighLevel_cfi import *
exoticaHTHLT = hltHighLevel
#Define the HLT path to be used.
exoticaHTHLT.HLTPaths =['HLT_HT100U']
exoticaHTHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29")

#Define the HLT quality cut 

exoticaHLTHTFilter = cms.EDFilter("HLTSummaryFilter",
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29"),
    member  = cms.InputTag("hltJet15UHt","","HLT8E29"),				
    cut     = cms.string("pt>250"),                     
    minN    = cms.int32(1)                  
)
                               
#Define the Reco quality cut
exoticaRecoHTFilter = cms.EDFilter("HLTGlobalSumsMET",
    inputTag = cms.InputTag("htMetKT4"),
	saveTag = cms.untracked.bool( True ),							  
	observable = cms.string( "sumEt" ),							  
    Min = cms.double(250.0),  
	Max = cms.double( -1.0 ),							  
    MinN = cms.int32(1)
)
## exoticaRecoHTFilter = cms.EDFilter("PATMETSelector",
##      src = cms.InputTag("htMetKT4"),
##      cut = cms.string("sumEt > 250.0"),
##      filter = cms.bool(True)
##  )


#Define group sequence, using HLT bits + either HLT/Reco quality cut. 
exoticaHTHLTQualitySeq = cms.Sequence(
   exoticaHTHLT+exoticaHLTHTFilter
   
)
exoticaHTRecoQualitySeq = cms.Sequence(
	exoticaHTHLT+exoticaRecoHTFilter
)
