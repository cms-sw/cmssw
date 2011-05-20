import FWCore.ParameterSet.Config as cms

#Define the HLT path to be used.
import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
exoticaDiMuonHLT = hlt.hltHighLevel.clone()
exoticaDiMuonHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT") 
exoticaDiMuonHLT.HLTPaths =['HLT_L2DoubleMu23_NoVertex_v*']

exoticaDiElectronHLT = hlt.hltHighLevel.clone()
exoticaDiElectronHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
exoticaDiElectronHLT.HLTPaths =['HLT_DoubleEle10_CaloIdL_TrkIdVL_Ele10_v*']

exoticaEMuHLT = hlt.hltHighLevel.clone()
exoticaEMuHLT.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
exoticaEMuHLT.HLTPaths =['HLT_Mu8_Ele17_CaloIdL_v*','HLT_Mu17_Ele8_CaloIdL_v*']

#Define the Reco quality cut
exoticaRecoDiMuonFilter = cms.EDFilter(
    "PtMinMuonCountFilter",
    src = cms.InputTag("muons"),
    ptMin = cms.double(25.0),
    minNumber = cms.uint32(2)
    )
exoticaRecoDiElectronFilter = cms.EDFilter(
    "PtMinGsfElectronCountFilter",
    src = cms.InputTag("gsfElectrons"),
    ptMin = cms.double(12.0),
    minNumber = cms.uint32(2)
    )
exoticaRecoMuonFilter = cms.EDFilter(
    "PtMinMuonCountFilter",
    src = cms.InputTag("muons"),
    ptMin = cms.double(10.0),
    minNumber = cms.uint32(1)
    )
exoticaRecoElectronFilter = cms.EDFilter(
    "PtMinGsfElectronCountFilter",
    src = cms.InputTag("gsfElectrons"),
    ptMin = cms.double(10.0),
    minNumber = cms.uint32(1)
    )

#
exoDiMuSequence = cms.Sequence(
    exoticaDiMuonHLT+exoticaRecoDiMuonFilter
)
exoDiEleSequence = cms.Sequence(
    exoticaDiElectronHLT+exoticaRecoDiElectronFilter
)

exoEMuSequence = cms.Sequence(
    exoticaEMuHLT+exoticaRecoElectronFilter+exoticaRecoMuonFilter
)

