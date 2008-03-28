import FWCore.ParameterSet.Config as cms

from SUSYBSMAnalysis.Skimming.SusyHLTPaths_cff import *
photonFilter2x20 = cms.EDFilter("EtMinPhotonCountFilter",
    src = cms.InputTag("photons"),
    etMin = cms.double(20.0),
    minNumber = cms.uint32(2)
)

photonFilter1x80 = cms.EDFilter("EtMinPhotonCountFilter",
    src = cms.InputTag("photons"),
    etMin = cms.double(80.0),
    minNumber = cms.uint32(1)
)

susyElectron = cms.Path(susyHLTElectronPath)
susyPhoton = cms.Path(susyHLTPhotonPath+photonFilter2x20+photonFilter1x80)

