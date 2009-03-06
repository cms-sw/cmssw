import FWCore.ParameterSet.Config as cms

#
# keep potential top spaecific default replacements here
#

#---------------------------------------
# Electron
#---------------------------------------
from PhysicsTools.PatAlgos.cleaningLayer0.electronCleaner_cfi import allLayer0Electrons

#---------------------------------------
# Muon
#---------------------------------------
from PhysicsTools.PatAlgos.cleaningLayer0.muonCleaner_cfi import allLayer0Muons

#---------------------------------------
# Tau
#---------------------------------------
from PhysicsTools.PatAlgos.cleaningLayer0.pfTauCleaner_cfi import allLayer0Taus

#---------------------------------------
# Jet
#---------------------------------------
from PhysicsTools.PatAlgos.cleaningLayer0.caloJetCleaner_cfi import allLayer0Jets

#---------------------------------------
# MET
#---------------------------------------
from PhysicsTools.PatAlgos.cleaningLayer0.caloMetCleaner_cfi import allLayer0METs

