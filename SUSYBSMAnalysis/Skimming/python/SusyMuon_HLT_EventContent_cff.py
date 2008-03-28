import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
AODSIMSusyHLTMuonEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
susyHLTMuonEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring()
    )
)
AODSIMSusyHLTMuonEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
susyHLTMuonEventSelection.SelectEvents.SelectEvents.append('susyMuon')

