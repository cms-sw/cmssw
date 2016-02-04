import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from TopQuarkAnalysis.TopSkimming.topDiLepton2Electron_EventContent_cff import *
topDiLepton2ElectronAODSIMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
topDiLepton2ElectronAODSIMEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
topDiLepton2ElectronAODSIMEventContent.outputCommands.extend(topDiLepton2ElectronEventContent.outputCommands)

