import FWCore.ParameterSet.Config as cms

topLeptonTauEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
topLeptonTauETauEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('topLeptonTauETauPath')
    )
)
topLeptonTauMuTauEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('topLeptonTauMuTauPath')
    )
)
topLeptonTauEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('topLeptonTauETauPath', 
            'topLeptonTauMuTauPath')
    )
)

