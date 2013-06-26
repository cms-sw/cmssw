import FWCore.ParameterSet.Config as cms

from HLTrigger.HLTfilters.hltHighLevel_cfi import *
exoticaSingleJetHLT = hltHighLevel
#Define the HLT path to be used.
exoticaSingleJetHLT.HLTPaths =['HLT_Jet50U']
exoticaSingleJetHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29")
#Define the HLT quality cut 

exoticaHLTSingleJetFilter = cms.EDFilter("HLTSummaryFilter",
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29"), # trigger summary
#    member  = cms.InputTag("hlt1jet30","","HLT"),      # filter
    member  = cms.InputTag("hltMCJetCorJetIcone5HF07","","HLT8E29"),  # or collection
    cut     = cms.string("pt>100"),                     # cut on trigger object
    minN    = cms.int32(1)                  # min. # of passing objects needed
 )
                               
#Define the Reco quality cut
exoticaRecoSingleJetFilter = cms.EDFilter("CaloJetSelector",
    src = cms.InputTag("antikt5CaloJets"),
    cut = cms.string("pt > 100.0"),
    filter = cms.bool(True)
)

#Define group sequence, using HLT bits + either HLT/Reco quality cut. 
exoticaSingleJetHLTQualitySeq = cms.Sequence(
   exoticaSingleJetHLT + exoticaHLTSingleJetFilter
   
)
exoticaSingleJetRecoQualitySeq = cms.Sequence(
	exoticaSingleJetHLT + exoticaRecoSingleJetFilter
)

