import FWCore.ParameterSet.Config as cms

from HLTrigger.HLTfilters.hltHighLevel_cfi import *
exoticaMuHLT = hltHighLevel
#Define the HLT path to be used.
exoticaMuHLT.HLTPaths =['HLT_Mu9']
exoticaMuHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29")

#Define the HLT quality cut 
exoticaHLTMuonFilter = cms.EDFilter("HLTSummaryFilter",
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29"), # trigger summary
    member  = cms.InputTag("hltL3MuonCandidates","","HLT8E29"),      # filter or collection									
    cut     = cms.string("pt>20"),                     # cut on trigger object
    minN    = cms.int32(1)                  # min. # of passing objects needed
 )
                               

#Define the Reco quality cut
exoticaRecoMuonFilter = cms.EDFilter("MuonRefSelector",
	src = cms.InputTag("muons"),
    cut = cms.string('pt > 20.0'),
    filter = cms.bool(True)            
                                      
)

#Define group sequence, using HLT/Reco quality cut. 
exoticaMuHLTQualitySeq = cms.Sequence(
	exoticaMuHLT+exoticaHLTMuonFilter
)
exoticaMuRecoQualitySeq = cms.Sequence(
    exoticaMuHLT+exoticaRecoMuonFilter
)

