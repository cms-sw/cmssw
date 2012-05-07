import FWCore.ParameterSet.Config as cms

from TopQuarkAnalysis.TopSkimming.topDiLepton2Electron_HLTSequences_cff import *
from TopQuarkAnalysis.TopSkimming.topDiLepton2Electron_Sequences_cff import *
topDiLepton2ElectronPath = cms.Path(topDiLepton2ElectronHLT+topDiLepton2Electron)

