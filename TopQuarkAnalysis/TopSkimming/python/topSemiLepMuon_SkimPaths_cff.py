import FWCore.ParameterSet.Config as cms

from TopQuarkAnalysis.TopSkimming.topSemiLepMuon_HLTSequences_cff import *
from TopQuarkAnalysis.TopSkimming.topSemiLepMuon_Sequences_cff import *
topSemiLepMuonPlus1JetPath = cms.Path(topSemiLepMuonHLT+topSemiLepMuonPlus1Jet)

