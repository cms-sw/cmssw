
import FWCore.ParameterSet.Config as cms

from HLTrigger.HLTfilters.hltHighLevel_cfi import *
exoticaDiPhoHLT = hltHighLevel
#Define the HLT path to be used.
exoticaDiPhoHLT.HLTPaths =['HLT_DoublePhoton10_L1R']
exoticaDiPhoHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29")
#Define the HLT quality cut 
exoticaHLTDiPhoFilter = cms.EDFilter("HLTSummaryFilter",
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29"), # trigger summary
    member  = cms.InputTag("hltL1NonIsoRecoEcalCandidate","","HLT8E29"),      # filter or collection
    cut     = cms.string("pt>20"),                     # cut on trigger object
    minN    = cms.int32(2)                  # min. # of passing objects needed
 )
                               
#Define the Reco quality cut
exoticaRecoDiPhoFilter = cms.EDFilter("EtMinPhotonCountFilter",
    src = cms.InputTag("photons"),
    etMin   = cms.double(20.),                    
    minNumber = cms.uint32(2)									   
)

#Define group sequence, using HLT/Reco quality cut. 
exoticaDiPhoHLTQualitySeq = cms.Sequence(
	exoticaDiPhoHLT+exoticaHLTDiPhoFilter
)
exoticaDiPhoRecoQualitySeq = cms.Sequence(
    exoticaDiPhoHLT+exoticaRecoDiPhoFilter
)


