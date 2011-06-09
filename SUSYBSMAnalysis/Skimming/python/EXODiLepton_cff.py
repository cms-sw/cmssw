import FWCore.ParameterSet.Config as cms

#Define the HLT path to be used.
import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
exoticaDiMuonHLT = hlt.hltHighLevel.clone()
exoticaDiMuonHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT") 
exoticaDiMuonHLT.HLTPaths = cms.vstring('HLT_L2DoubleMu23_NoVertex_v*')

exoticaDiElectronHLT = hlt.hltHighLevel.clone()
exoticaDiElectronHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
exoticaDiElectronHLT.HLTPaths = cms.vstring('HLT_DoubleEle10_CaloIdL_TrkIdVL_Ele10_v*')

exoticaDiEMuHLT = hlt.hltHighLevel.clone()
exoticaDiEMuHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
exoticaDiEMuHLT.HLTPaths = cms.vstring('HLT_Mu8_Ele17_CaloIdL_v*','HLT_Mu17_Ele8_CaloIdL_v*')

#Define the Reco quality cut
exoticaRecoDiDiMuonFilter = cms.EDFilter(
    "PtMinMuonCountFilter",
    src = cms.InputTag("muons"),
    ptMin = cms.double(25.0),
    minNumber = cms.uint32(2)
    )
exoticaRecoDiDiElectronFilter = cms.EDFilter(
    "PtMinGsfElectronCountFilter",
    src = cms.InputTag("gsfElectrons"),
    ptMin = cms.double(12.0),
    minNumber = cms.uint32(2)
    )
exoticaRecoDiSiMuonFilter = cms.EDFilter(
    "PtMinMuonCountFilter",
    src = cms.InputTag("muons"),
    ptMin = cms.double(10.0),
    minNumber = cms.uint32(1)
    )
exoticaRecoDiSiElectronFilter = cms.EDFilter(
    "PtMinGsfElectronCountFilter",
    src = cms.InputTag("gsfElectrons"),
    ptMin = cms.double(10.0),
    minNumber = cms.uint32(1)
    )

#
exoDiMuSequence = cms.Sequence(
    exoticaDiMuonHLT+exoticaRecoDiDiMuonFilter
)
exoDiEleSequence = cms.Sequence(
    exoticaDiElectronHLT+exoticaRecoDiDiElectronFilter
)

exoEMuSequence = cms.Sequence(
    exoticaDiEMuHLT+exoticaRecoDiSiElectronFilter+exoticaRecoDiSiMuonFilter
)

