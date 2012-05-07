import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from TopQuarkAnalysis.TopSkimming.topFullyHadronic_EventContent_cff import *
topFullyHadronicAODSIMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
topFullyHadronicAODSIMEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
topFullyHadronicAODSIMEventContent.outputCommands.extend(topFullyHadronicEventContent.outputCommands)

