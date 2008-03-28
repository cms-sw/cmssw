import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
AODSIMSusyHLTElectronPhotonEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
susyHLTElectronPhotonEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring()
    )
)
AODSIMSusyHLTElectronPhotonEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
susyHLTElectronPhotonEventSelection.SelectEvents.SelectEvents.append('susyPhoton')
susyHLTElectronPhotonEventSelection.SelectEvents.SelectEvents.append('susyElectron')

