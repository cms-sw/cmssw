import FWCore.ParameterSet.Config as cms

from TopQuarkAnalysis.TopSkimming.topSemiLepElectron_HLTSequences_cff import *
from TopQuarkAnalysis.TopSkimming.topSemiLepElectron_Sequences_cff import *
topSemiLepElectronPlus1JetPath = cms.Path(topSemiLepElectronHLT+topSemiLepElectronPlus1Jet)

