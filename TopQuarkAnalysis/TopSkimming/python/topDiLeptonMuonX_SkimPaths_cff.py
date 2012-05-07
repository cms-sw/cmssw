import FWCore.ParameterSet.Config as cms

from TopQuarkAnalysis.TopSkimming.topDiLeptonMuonX_HLTSequences_cff import *
from TopQuarkAnalysis.TopSkimming.topDiLeptonMuonX_Sequences_cff import *
topDiLeptonMuonXPath = cms.Path(topDiLeptonMuonXHLT+topDiLeptonMuonX)

