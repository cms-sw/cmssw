import FWCore.ParameterSet.Config as cms

#---------------------------------------
# Electron
#---------------------------------------
from PhysicsTools.PatAlgos.mcMatchLayer0.electronMatch_cfi import electronMatch

electronMatch.src         = "allLayer0Electrons"  ## input source
electronMatch.matched     = "genParticles"        ## match to
electronMatch.maxDeltaR   = 0.5                   ## Minimum deltaR for the match
electronMatch.maxDPtRel   = 0.5                   ## Minimum deltaPt/Pt for the match
electronMatch.resolveAmbiguities    = True        ## Forbid two RECO objects to match to the same GEN object
electronMatch.resolveByMatchQuality = False       ## False = just match input in pt order;
                                                  ## True  = pick lowest deltaR pair first
electronMatch.checkCharge = True                  ## True = require RECO and MC objects to have the same charge
electronMatch.mcPdgId     = [11]                  ## one or more PDG ID (11 = electron); absolute values
electronMatch.mcStatus    = [ 1]                  ## PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)


#---------------------------------------
# Muon
#---------------------------------------
from PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi import muonMatch

muonMatch.src             = "allLayer0Muons"      ## input source
muonMatch.matched         = "genParticles"        ## match to
muonMatch.maxDeltaR       = 0.5                   ## Minimum deltaR for the match
muonMatch.maxDPtRel       = 0.5                   ## Minimum deltaPt/Pt for the match
muonMatch.resolveAmbiguities    = True            ## Forbid two RECO objects to match to the same GEN object
muonMatch.resolveByMatchQuality = False           ## False = just match input in pt order;
                                                  ## True  = pick lowest deltaR pair first
muonMatch.checkCharge     = True                  ## True = require RECO and MC objects to have the same charge
muonMatch.mcPdgId         = [13]                  ## one or more PDG ID (13 = muon); absolute values
muonMatch.mcStatus        = [ 1]                  ## PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)


#---------------------------------------
# Tau
#---------------------------------------
from PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi import tauMatch

tauMatch.src              = "allLayer0Taus"      ## input source
tauMatch.matched          = "genParticles"       ## match to
tauMatch.maxDeltaR        = 5.0                  ## Minimum deltaR for the match
tauMatch.maxDPtRel        = 99.                  ## Minimum deltaPt/Pt for the match
tauMatch.resolveAmbiguities    = True            ## Forbid two RECO objects to match to the same GEN object
tauMatch.resolveByMatchQuality = False           ## False = just match input in pt order;
                                                 ## True  = pick lowest deltaR pair first
tauMatch.checkCharge      = True                 ## True = require RECO and MC objects to have the same charge
tauMatch.mcPdgId          = [15]                 ## one or more PDG ID (15 = tau); absolute values
tauMatch.mcStatus         = [ 2]                 ## PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)


#---------------------------------------
# Jet (parton match)
#---------------------------------------
from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi import jetPartonMatch

jetPartonMatch.src        = "allLayer0Jets"      ## input source
jetPartonMatch.matched    = "genParticles"       ## match to
jetPartonMatch.maxDeltaR  = 0.5                  ## Minimum deltaR for the match
jetPartonMatch.maxDPtRel  = 5.0                  ## Minimum deltaPt/Pt for the match
jetPartonMatch.resolveAmbiguities    = True      ## Forbid two RECO objects to match to the same GEN object
jetPartonMatch.resolveByMatchQuality = False     ## False = just match input in pt order;
                                                 ## True  = pick lowest deltaR pair first
jetPartonMatch.checkCharge= False                ## True = require RECO and MC objects to have the same charge
jetPartonMatch.mcPdgId    = [1, 2, 3, 4, 5, 21]  ## one or more PDG ID (15 = tau); absolute values
jetPartonMatch.mcStatus   = [2]                  ## PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering


#---------------------------------------
# Jet (genJet match)
#---------------------------------------
from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi import jetGenJetMatch

jetGenJetMatch.src        = "allLayer0Jets"      ## input source
jetGenJetMatch.matched    = "iterativeCone5GenJets"
jetGenJetMatch.maxDeltaR  = 0.5                  ## Minimum deltaR for the match
jetGenJetMatch.maxDPtRel  = 5.0                  ## Minimum deltaPt/Pt for the match
jetGenJetMatch.resolveAmbiguities    = True      ## Forbid two RECO objects to match to the same GEN object
jetGenJetMatch.resolveByMatchQuality = False     ## False = just match input in order;
                                                 ## True  = pick lowest deltaR pair first
jetGenJetMatch.checkCharge= False                ## True = require RECO and MC objects to have the same charge
jetGenJetMatch.mcPdgId    = []                   ##
jetGenJetMatch.mcStatus   = []                   ##

