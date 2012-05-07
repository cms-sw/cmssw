import FWCore.ParameterSet.Config as cms

from TopQuarkAnalysis.TopSkimming.topLeptonTau_Sequences_cff import *
topLeptonTauMuTauPath = cms.Path(topLeptonTauMuTau)
topLeptonTauETauPath = cms.Path(topLeptonTauETau)

