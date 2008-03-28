import FWCore.ParameterSet.Config as cms

from SUSYBSMAnalysis.Skimming.SusyHLTPaths_cff import *
jetFilter2x30 = cms.EDFilter("EtMinCaloJetCountFilter",
    src = cms.InputTag("iterativeCone5CaloJets"),
    etMin = cms.double(30.0),
    minNumber = cms.uint32(2)
)

jetFilter1x80 = cms.EDFilter("EtMinCaloJetCountFilter",
    src = cms.InputTag("iterativeCone5CaloJets"),
    etMin = cms.double(80.0),
    minNumber = cms.uint32(1)
)

susyJetMET = cms.Path(susyHLTJetMETPath+jetFilter2x30+jetFilter1x80)
susyMETOnly = cms.Path(susyHLTMETOnlyPath)

