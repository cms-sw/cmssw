import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from TopQuarkAnalysis.TopSkimming.topSemiLepElectron_EventContent_cff import *
topSemiLepElectronAODSIMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
topSemiLepElectronAODSIMEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
topSemiLepElectronAODSIMEventContent.outputCommands.extend(topSemiLepElectronEventContent.outputCommands)

