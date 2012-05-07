import FWCore.ParameterSet.Config as cms

topDiLeptonMuonXFilter = cms.EDFilter("TopDiLeptonFilter",
    electronCollection = cms.InputTag("pixelMatchGsfElectrons"),
    muonCollection = cms.InputTag("muons"),
    ptThreshold = cms.double(20.0)
)

topDiLeptonMuonX = cms.Sequence(topDiLeptonMuonXFilter)

