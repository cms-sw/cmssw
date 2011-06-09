import FWCore.ParameterSet.Config as cms

#Define the HLT path to be used.
import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
exoticaTriMuonHLT = hlt.hltHighLevel.clone()
exoticaTriMuonHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
exoticaTriMuonHLT.HLTPaths = cms.vstring('HLT_TripleMu*')

exoticaTriElectronHLT = hlt.hltHighLevel.clone()
exoticaTriElectronHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
exoticaTriElectronHLT.HLTPaths = cms.vstring('HLT_Mu5_Ele8_CaloIdL_TrkIdVL_Ele8_v*','HLT_DoubleMu5_Ele8_v*')


exoticaEMuHLT = hlt.hltHighLevel.clone()
exoticaEMuHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
exoticaEMuHLT.HLTPaths =cms.vstring('HLT_DoubleEle5_SW_L1R','HLT_DoubleMu3_v*')
exoticaEMuHLT.andOr = cms.bool( True )
exoticaEMuHLT.throw = cms.bool( False )

#Define the Reco quality cut
exoticaRecoTriTriMuonFilter = cms.EDFilter(
    "PtMinMuonCountFilter",
    src = cms.InputTag("muons"),
    ptMin = cms.double(5.0),
    minNumber = cms.uint32(3)
)
exoticaRecoTriTriElectronFilter = cms.EDFilter(
    "PtMinGsfElectronCountFilter",
    src = cms.InputTag("gsfElectrons"),
    ptMin = cms.double(5.0),
    minNumber = cms.uint32(3)
    )
exoticaRecoTriDiMuonFilter = cms.EDFilter(
    "PtMinMuonCountFilter",
    src = cms.InputTag("muons"),
    ptMin = cms.double(10.0),		   
    minNumber = cms.uint32(2)
)
exoticaRecoTriDiElectronFilter = cms.EDFilter(
    "PtMinGsfElectronCountFilter",
    src = cms.InputTag("gsfElectrons"),
    ptMin = cms.double(5.0),
    minNumber = cms.uint32(2)		   
)
exoticaRecoTriSiMuonFilter = cms.EDFilter(
    "PtMinMuonCountFilter",
    src = cms.InputTag("muons"),
    ptMin = cms.double(5.0),   		     
    minNumber = cms.uint32(1)
)
exoticaRecoTriSiElectronFilter = cms.EDFilter(
    "PtMinGsfElectronCountFilter",
    src = cms.InputTag("gsfElectrons"),
    ptMin = cms.double(10.0), 		   
    minNumber = cms.uint32(1)		   
    )

#
exoTriMuonSequence = cms.Sequence(
    exoticaTriMuonHLT+exoticaRecoTriTriMuonFilter
)
exoTriElectronSequence = cms.Sequence(
    exoticaTriElectronHLT+exoticaRecoTriTriElectronFilter
)

exo1E2MuSequence = cms.Sequence(
    exoticaEMuHLT+exoticaRecoTriSiElectronFilter+exoticaRecoTriDiMuonFilter
)
exo2E1MuSequence = cms.Sequence(
    exoticaEMuHLT+exoticaRecoTriDiElectronFilter+exoticaRecoTriSiMuonFilter
)
