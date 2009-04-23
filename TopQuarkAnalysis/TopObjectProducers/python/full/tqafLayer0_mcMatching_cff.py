import FWCore.ParameterSet.Config as cms

#
# keep potential top spaecific default replacements here
#

#---------------------------------------
# Electron
#---------------------------------------
from PhysicsTools.PatAlgos.mcMatchLayer0.electronMatch_cfi import electronMatch

#---------------------------------------
# Muon
#---------------------------------------
from PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi     import muonMatch

#---------------------------------------
# Tau
#---------------------------------------
from PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi      import tauMatch

#---------------------------------------
# Jet (parton match)
#---------------------------------------
from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi      import jetPartonMatch

#---------------------------------------
# Jet (genJet match)
#---------------------------------------
from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi      import jetGenJetMatch

