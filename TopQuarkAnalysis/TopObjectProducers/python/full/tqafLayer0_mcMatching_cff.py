import FWCore.ParameterSet.Config as cms

#---------------------------------------
# Electron
#---------------------------------------
from PhysicsTools.PatAlgos.mcMatchLayer0.electronMatch_cfi import electronMatch

electronMatch.src = cms.InputTag("allLayer0Electrons")  ## input source
electronMatch.matched   = cms.InputTag("genParticles")  ## match to
electronMatch.maxDeltaR = cms.double(0.5)               ## Minimum deltaR for the match
electronMatch.maxDPtRel = cms.double(0.5)               ## Minimum deltaPt/Pt for the match
electronMatch.resolveAmbiguities = cms.bool(True)       ## Forbid two RECO objects to match to the same GEN object
electronMatch.resolveByMatchQuality = cms.bool(True)    ## False = just match input in order;
                                                        ## True  = pick lowest deltaR pair first
electronMatch.checkCharge = cms.bool(True)              ## True = require RECO and MC objects to have the same charge
electronMatch.mcPdgId  = cms.vint32(11)                 ## one or more PDG ID (11 = electron); absolute values
electronMatch.mcStatus = cms.vint32(1)                  ## PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)


#---------------------------------------
# Muon
#---------------------------------------
from PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi import muonMatch

muonMatch.src = cms.InputTag("allLayer0Muons")          ## input source
muonMatch.matched   = cms.InputTag("genParticles")      ## match to
muonMatch.maxDeltaR = cms.double(0.5)                   ## Minimum deltaR for the match
muonMatch.maxDPtRel = cms.double(0.5)                   ## Minimum deltaPt/Pt for the match
muonMatch.resolveAmbiguities = cms.bool(True)           ## Forbid two RECO objects to match to the same GEN object
muonMatch.resolveByMatchQuality = cms.bool(True)        ## False = just match input in order;
                                                        ## True  = pick lowest deltaR pair first
muonMatch.checkCharge = cms.bool(True)                  ## True = require RECO and MC objects to have the same charge
muonMatch.mcPdgId  = cms.vint32(13)                     ## one or more PDG ID (13 = muon); absolute values
muonMatch.mcStatus = cms.vint32(1)                      ## PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)


#---------------------------------------
# Tau
#---------------------------------------
from PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi import tauMatch

tauMatch.src = cms.InputTag("allLayer0Taus")            ## input source
tauMatch.matched   = cms.InputTag("genParticles")       ## match to
tauMatch.maxDeltaR = cms.double(5.0)                    ## Minimum deltaR for the match
tauMatch.maxDPtRel = cms.double(99.)                    ## Minimum deltaPt/Pt for the match
tauMatch.resolveAmbiguities = cms.bool(True)            ## Forbid two RECO objects to match to the same GEN object
tauMatch.resolveByMatchQuality = cms.bool(True)         ## False = just match input in order;
                                                        ## True  = pick lowest deltaR pair first
tauMatch.checkCharge = cms.bool(True)                   ## True = require RECO and MC objects to have the same charge
tauMatch.mcPdgId  = cms.vint32(15)                      ## one or more PDG ID (15 = tau); absolute values
tauMatch.mcStatus = cms.vint32(2)                       ## PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)


#---------------------------------------
# Jet (parton match)
#---------------------------------------
from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi import jetPartonMatch

jetPartonMatch.src = cms.InputTag("allLayer0Jets")      ## input source
jetPartonMatch.matched   = cms.InputTag("genParticles") ## match to
jetPartonMatch.maxDeltaR = cms.double(0.5)              ## Minimum deltaR for the match
jetPartonMatch.maxDPtRel = cms.double(5.0)              ## Minimum deltaPt/Pt for the match
jetPartonMatch.resolveAmbiguities = cms.bool(True)      ## Forbid two RECO objects to match to the same GEN object
jetPartonMatch.resolveByMatchQuality = cms.bool(False)  ## False = just match input in order;
                                                        ## True  = pick lowest deltaR pair first
jetPartonMatch.checkCharge = cms.bool(False)            ## True = require RECO and MC objects to have the same charge
jetPartonMatch.mcPdgId  = cms.vint32(1, 2, 3, 4, 5, 21) ## one or more PDG ID (15 = tau); absolute values
jetPartonMatch.mcStatus = cms.vint32(2)                 ## PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering


#---------------------------------------
# Jet (genJet match)
#---------------------------------------
from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi import jetGenJetMatch

jetGenJetMatch.src = cms.InputTag("allLayer0Jets")      ## input source
jetGenJetMatch.matched   = cms.InputTag("iterativeCone5GenJets")
jetGenJetMatch.maxDeltaR = cms.double(0.4)              ## Minimum deltaR for the match
jetGenJetMatch.maxDPtRel = cms.double(3.0)              ## Minimum deltaPt/Pt for the match
jetGenJetMatch.resolveAmbiguities = cms.bool(True)      ## Forbid two RECO objects to match to the same GEN object
jetGenJetMatch.resolveByMatchQuality = cms.bool(False)  ## False = just match input in order;
                                                        ## True  = pick lowest deltaR pair first
jetGenJetMatch.checkCharge = cms.bool(False)            ## True = require RECO and MC objects to have the same charge
jetGenJetMatch.mcPdgId  = cms.vint32()                  ##
jetGenJetMatch.mcStatus = cms.vint32()                  ##

