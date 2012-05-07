import FWCore.ParameterSet.Config as cms

from TopQuarkAnalysis.TopSkimming.topFullyHadronic_HLTSequences_cff import *
from TopQuarkAnalysis.TopSkimming.topFullyHadronic_Sequences_cff import *
topFullyHadronicJetsPath = cms.Path(topFullyHadronicBJetsHLT+topFullyHadronicJets)
topFullyHadronicBJetsPath = cms.Path(topFullyHadronicBJetsHLT+topFullyHadronicBJets)

