import FWCore.ParameterSet.Config as cms

#Define the HLT path to be used.
import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
exoticaTriMuonHLT = hlt.hltHighLevel.clone()
exoticaTriMuonHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
exoticaTriMuonHLT.HLTPaths =['HLT_TripleMu*']

exoticaTriElectronHLT = hlt.hltHighLevel.clone()
exoticaTriElectronHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
exoticaTriElectronHLT.HLTPaths =['HLT_Mu5_Ele8_CaloIdL_TrkIdVL_Ele8_v*','HLT_DoubleMu5_Ele8_v*']


exoticaEMuHLT = hlt.hltHighLevel.clone()
exoticaEMuHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
exoticaEMuHLT.HLTPaths =['HLT_DoubleEle5_SW_L1R','HLT_DoubleMu3_v*']
exoticaEMuHLT.andOr = cms.bool( True )
exoticaEMuHLT.throw = cms.bool( False )

#Define the Reco quality cut
exoticaRecoTriMuonFilter = cms.EDFilter(
    "PtMinMuonCountFilter",
    src = cms.InputTag("muons"),
    ptMin = cms.double(5.0),
    minNumber = cms.uint32(3)
)
exoticaRecoTriElectronFilter = cms.EDFilter(
    "PtMinGsfElectronCountFilter",
    src = cms.InputTag("gsfElectrons"),
    ptMin = cms.double(5.0),
    minNumber = cms.uint32(3)
    )
exoticaRecoDiMuonFilter = cms.EDFilter(
    "PtMinMuonCountFilter",
    src = cms.InputTag("muons"),
    ptMin = cms.double(10.0),		   
    minNumber = cms.uint32(2)
)
exoticaRecoDiElectronFilter = cms.EDFilter(
    "PtMinGsfElectronCountFilter",
    src = cms.InputTag("gsfElectrons"),
    ptMin = cms.double(5.0),
    minNumber = cms.uint32(2)		   
)
exoticaRecoMuonFilter = cms.EDFilter(
    "PtMinMuonCountFilter",
    src = cms.InputTag("muons"),
    ptMin = cms.double(5.0),   		     
    minNumber = cms.uint32(1)
)
exoticaRecoElectronFilter = cms.EDFilter(
    "PtMinGsfElectronCountFilter",
    src = cms.InputTag("gsfElectrons"),
    ptMin = cms.double(10.0), 		   
    minNumber = cms.uint32(1)		   
    )

#
exoTriMuonSequence = cms.Sequence(
    exoticaTriMuonHLT+exoticaRecoTriMuonFilter
)
exoTriElectronSequence = cms.Sequence(
    exoticaTriElectronHLT+exoticaRecoTriElectronFilter
)

exo1E2MuSequence = cms.Sequence(
    exoticaEMuHLT+exoticaRecoElectronFilter+exoticaRecoDiMuonFilter
)
exo2E1MuSequence = cms.Sequence(
    exoticaEMuHLT+exoticaRecoDiElectronFilter+exoticaRecoMuonFilter
)
