import FWCore.ParameterSet.Config as cms

topDiLepton2ElectronFilter = cms.EDFilter("TopDiLeptonFilter",
    electronCollection = cms.InputTag("pixelMatchGsfElectrons"),
    muonCollection = cms.InputTag("muons"),
    ptThreshold = cms.double(20.0)
)

topDiLepton2Electron = cms.Sequence(topDiLepton2ElectronFilter)

