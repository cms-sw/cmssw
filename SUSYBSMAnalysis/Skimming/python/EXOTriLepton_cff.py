
import FWCore.ParameterSet.Config as cms

from HLTrigger.HLTfilters.hltHighLevel_cfi import *


#Define the HLT path to be used.
exoticaTriMuonHLT = hltHighLevel.clone()
exoticaTriMuonHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29")
exoticaTriMuonHLT.HLTPaths =['HLT_DoubleMu3']

exoticaTriElectronHLT = hltHighLevel.clone()
exoticaTriElectronHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29")
exoticaTriElectronHLT.HLTPaths =['HLT_DoubleEle5_SW_L1R']


exoticaEMuHLT = hltHighLevel.clone()
exoticaEMuHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29")
exoticaEMuHLT.HLTPaths =['HLT_DoubleEle5_SW_L1R','HLT_DoubleMu3']


#Define the HLT quality cut 
from HLTrigger.HLTfilters.hltSummaryFilter_cfi import *
exoticaHLTTriMuonFilter = hltSummaryFilter.clone(
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29"), # trigger summary
    member  = cms.InputTag("hltL3MuonCandidates","","HLT8E29"),      # filter or collection
    cut     = cms.string("pt>5"),                     # cut on trigger object
    minN    = cms.int32(3)                  # min. # of passing objects needed
 )
                               
exoticaHLTTriElectronFilter =hltSummaryFilter.clone(										  
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29"), # trigger summary
    member  = cms.InputTag("hltL1NonIsoRecoEcalCandidate","","HLT8E29"),      # filter or collection
    #member  = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt5PixelMatchFilter","","HLT8E29"),      # filter or collection
    cut     = cms.string("pt>5"),                     # cut on trigger object
    minN    = cms.int32(3)                  # min. # of passing objects needed
 )

exoticaHLTDiMuonFilter = hltSummaryFilter.clone(
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29"), # trigger summary
    member  = cms.InputTag("hltL3MuonCandidates","","HLT8E29"),      # filter or collection
    cut     = cms.string("pt>5"),                     # cut on trigger object
    minN    = cms.int32(2)                  # min. # of passing objects needed
 )
     
exoticaHLTDiElectronFilter =hltSummaryFilter.clone(										  
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29"), # trigger summary
    member  = cms.InputTag("hltL1NonIsoRecoEcalCandidate","","HLT8E29"),      # filter or collection
#    member  = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt5PixelMatchFilter","","HLT8E29"),      # filter or collection
    cut     = cms.string("pt>5"),                     # cut on trigger object
    minN    = cms.int32(2)                  # min. # of passing objects needed
 )

exoticaHLTMuonFilter = hltSummaryFilter.clone(
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29"), # trigger summary
    member  = cms.InputTag("hltL3MuonCandidates","","HLT8E29"),      # filter or collection
    cut     = cms.string("pt>5"),                     # cut on trigger object
    minN    = cms.int32(1)                  # min. # of passing objects needed
 )
     
exoticaHLTElectronFilter =hltSummaryFilter.clone(										  
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT8E29"), # trigger summary
    member  = cms.InputTag("hltL1NonIsoRecoEcalCandidate","","HLT8E29"),      # filter or collection
 #   member  = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt5PixelMatchFilter","","HLT8E29"),      # filter or collection
    cut     = cms.string("pt>5"),                     # cut on trigger object
    minN    = cms.int32(1)                  # min. # of passing objects needed
 )




#Define the Reco quality cut
exoticaRecoTriMuonFilter = cms.EDFilter("PtMinMuonCountFilter",
	src = cms.InputTag("muons"),
    ptMin = cms.double(5.0),									   
    minNumber = cms.uint32(3)									   
)
exoticaRecoTriElectronFilter = cms.EDFilter("PtMinGsfElectronCountFilter",
	src = cms.InputTag("gsfElectrons"),
    ptMin = cms.double(5.0),									   
    minNumber = cms.uint32(3)									   
)
exoticaRecoDiMuonFilter = cms.EDFilter("PtMinMuonCountFilter",
	src = cms.InputTag("muons"),
    ptMin = cms.double(5.0),									   
    minNumber = cms.uint32(2)									   
)
exoticaRecoDiElectronFilter = cms.EDFilter("PtMinGsfElectronCountFilter",
	src = cms.InputTag("gsfElectrons"),
    ptMin = cms.double(5.0),									   
    minNumber = cms.uint32(2)									   
)
exoticaRecoMuonFilter = cms.EDFilter("PtMinMuonCountFilter",
	src = cms.InputTag("muons"),
    ptMin = cms.double(5.0),									   
    minNumber = cms.uint32(1)									   
)
exoticaRecoElectronFilter = cms.EDFilter("PtMinGsfElectronCountFilter",
	src = cms.InputTag("gsfElectrons"),
    ptMin = cms.double(5.0),									   
    minNumber = cms.uint32(1)									   
)


#Define group sequence, using HLT/Reco quality cut. 
exoticaTriMuonHLTQualitySeq = cms.Sequence(
	exoticaTriMuonHLT+exoticaHLTTriMuonFilter
)
exoticaTriElectronHLTQualitySeq = cms.Sequence(
	exoticaTriElectronHLT+exoticaHLTTriElectronFilter
)
exotica1E2MuHLTQualitySeq = cms.Sequence(
    exoticaEMuHLT+exoticaHLTElectronFilter+exoticaHLTDiMuonFilter
)
exotica2E1MuHLTQualitySeq = cms.Sequence(
    exoticaEMuHLT+exoticaHLTDiElectronFilter+exoticaHLTMuonFilter
)

#
exoticaTriMuonRecoQualitySeq = cms.Sequence(
    exoticaTriMuonHLT+exoticaRecoTriMuonFilter
)
exoticaTriElectronRecoQualitySeq = cms.Sequence(
    exoticaTriElectronHLT+exoticaRecoTriElectronFilter
)

exotica1E2MuRecoQualitySeq = cms.Sequence(
    exoticaEMuHLT+exoticaRecoElectronFilter+exoticaRecoDiMuonFilter
)
exotica2E1MuRecoQualitySeq = cms.Sequence(
    exoticaEMuHLT+exoticaRecoDiElectronFilter+exoticaRecoMuonFilter
)
