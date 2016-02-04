import FWCore.ParameterSet.Config as cms

topDiLepton2ElectronEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
topDiLepton2ElectronEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('topDiLepton2ElectronPath')
    )
)

