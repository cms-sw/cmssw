import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from TopQuarkAnalysis.TopSkimming.topLeptonTau_EventContent_cff import *
topLeptonTauAODSIMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
topLeptonTauAODSIMEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
topLeptonTauAODSIMEventContent.outputCommands.extend(topLeptonTauEventContent.outputCommands)

