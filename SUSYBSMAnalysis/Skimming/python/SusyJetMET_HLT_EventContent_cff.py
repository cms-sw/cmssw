import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
AODSIMSusyHLTJetMETEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
susyHLTJetMETEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring()
    )
)
AODSIMSusyHLTJetMETEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
susyHLTJetMETEventSelection.SelectEvents.SelectEvents.append('susyJetMET')
susyHLTJetMETEventSelection.SelectEvents.SelectEvents.append('susyMETOnly')

