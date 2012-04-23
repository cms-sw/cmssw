import FWCore.ParameterSet.Config as cms

topDiLeptonMuonXEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
topDiLeptonMuonXEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('topDiLeptonMuonXPath')
    )
)

