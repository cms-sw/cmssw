import FWCore.ParameterSet.Config as cms

topSemiLepMuonEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
topSemiLepMuonPlus1JetEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('topSemiLepMuonPlus1JetPath')
    )
)
topSemiLepMuonPlus2JetsEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('topSemiLepMuonPlus2JetsPath')
    )
)

