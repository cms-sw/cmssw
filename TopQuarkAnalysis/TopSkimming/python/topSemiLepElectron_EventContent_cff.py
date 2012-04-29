import FWCore.ParameterSet.Config as cms

topSemiLepElectronEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
topSemiLepElectronPlus1JetEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('topSemiLepElectronPlus1JetPath')
    )
)
topSemiLepElectronPlus2JetsEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('topSemiLepElectronPlus2JetsPath')
    )
)

