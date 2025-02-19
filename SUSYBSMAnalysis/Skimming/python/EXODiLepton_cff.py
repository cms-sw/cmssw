import FWCore.ParameterSet.Config as cms

from HLTrigger.HLTfilters.hltHighLevel_cfi import *
exoticaDiMuonHLT = hltHighLevel.clone()
#Define the HLT path to be used.
exoticaDiMuonHLT.HLTPaths =['HLT_DoubleMu3']
exoticaDiMuonHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29") 

exoticaDiElectronHLT = hltHighLevel.clone()
exoticaDiElectronHLT.HLTPaths =['HLT_Ele10_LW_L1R']
exoticaDiElectronHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29")

exoticaEMuHLT = hltHighLevel.clone()
exoticaEMuHLT.HLTPaths =['HLT_Ele10_LW_L1R','HLT_Mu5']
exoticaEMuHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29")

#Define the HLT quality cut 
from HLTrigger.HLTfilters.hltSummaryFilter_cfi import *

exoticaHLTDiMuonFilter = hltSummaryFilter.clone(
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29"), # trigger summary
    member  = cms.InputTag("hltL3MuonCandidates","","HLT8E29"),      # filter or collection
    cut     = cms.string("pt>10"),                     # cut on trigger object
    minN    = cms.int32(2)                  # min. # of passing objects needed
 )
                               
exoticaHLTDiElectronFilter =hltSummaryFilter.clone(										  
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29"), # trigger summary
    member  = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDphiFilter","","HLT8E29"),      # filter or collection
    cut     = cms.string("pt>10"),                     # cut on trigger object
    minN    = cms.int32(2)                  # min. # of passing objects needed
)

exoticaHLTMuonFilter = hltSummaryFilter.clone(
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29"), # trigger summary
    member  = cms.InputTag("hltL3MuonCandidates","","HLT8E29"),      # filter or collection
    cut     = cms.string("pt>10"),                     # cut on trigger object
    minN    = cms.int32(1)                  # min. # of passing objects needed
 )
   
exoticaHLTElectronFilter =hltSummaryFilter.clone(										  
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29"), # trigger summary
    member  = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDphiFilter","","HLT8E29"),      # filter or collection
    cut     = cms.string("pt>10"),                     # cut on trigger object
    minN    = cms.int32(1)                  # min. # of passing objects needed
 )


   

#Define the Reco quality cut
exoticaRecoDiMuonFilter = cms.EDFilter("PtMinMuonCountFilter",
	src = cms.InputTag("muons"),
    ptMin = cms.double(10.0),									   
    minNumber = cms.uint32(2)									   
)
exoticaRecoDiElectronFilter = cms.EDFilter("PtMinGsfElectronCountFilter",
	src = cms.InputTag("gsfElectrons"),
    ptMin = cms.double(10.0),									   
    minNumber = cms.uint32(2)									   
)
exoticaRecoMuonFilter = cms.EDFilter("PtMinMuonCountFilter",
	src = cms.InputTag("muons"),
    ptMin = cms.double(10.0),									   
    minNumber = cms.uint32(1)									   
)
exoticaRecoElectronFilter = cms.EDFilter("PtMinGsfElectronCountFilter",
	src = cms.InputTag("gsfElectrons"),
    ptMin = cms.double(10.0),									   
    minNumber = cms.uint32(1)									   
)


#Define group sequence, using HLT/Reco quality cut. 
exoticaDiMuonHLTQualitySeq = cms.Sequence(
	exoticaDiMuonHLT+exoticaHLTDiMuonFilter
)
exoticaDiElectronHLTQualitySeq = cms.Sequence(
	exoticaDiElectronHLT+exoticaHLTDiElectronFilter
)
exoticaEMuHLTQualitySeq = cms.Sequence(
    exoticaEMuHLT+exoticaHLTElectronFilter+exoticaHLTMuonFilter
)
#
exoticaDiMuonRecoQualitySeq = cms.Sequence(
    exoticaDiMuonHLT+exoticaRecoDiMuonFilter
)
exoticaDiElectronRecoQualitySeq = cms.Sequence(
    exoticaDiElectronHLT+exoticaRecoDiElectronFilter
)

exoticaEMuRecoQualitySeq = cms.Sequence(
    exoticaEMuHLT+exoticaRecoElectronFilter+exoticaRecoMuonFilter
)

