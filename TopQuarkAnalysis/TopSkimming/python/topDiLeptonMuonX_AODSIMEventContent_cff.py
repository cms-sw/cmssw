import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from TopQuarkAnalysis.TopSkimming.topDiLeptonMuonX_EventContent_cff import *
topDiLeptonMuonXAODSIMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
topDiLeptonMuonXAODSIMEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
topDiLeptonMuonXAODSIMEventContent.outputCommands.extend(topDiLeptonMuonXEventContent.outputCommands)

