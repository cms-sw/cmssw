import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from TopQuarkAnalysis.TopSkimming.topSemiLepMuon_EventContent_cff import *
topSemiLepMuonAODSIMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
topSemiLepMuonAODSIMEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
topSemiLepMuonAODSIMEventContent.outputCommands.extend(topSemiLepMuonEventContent.outputCommands)

