import FWCore.ParameterSet.Config as cms

#
# L0 input
#

## import module
from PhysicsTools.PatAlgos.cleaningLayer0.caloTauCleaner_cfi import allLayer0CaloTaus

## configure for tqaf
allLayer0CaloTaus.tauSource              = 'pfRecoTauProducer'
allLayer0CaloTaus.tauDiscriminatorSource = 'pfRecoTauDiscriminationByIsolation'

#
# genMatch
#

## import module
from PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi import tauMatch

## configure for tqaf
tauMatch.src                             = 'allLayer0Taus'
tauMatch.matched                         = 'genParticles'
tauMatch.maxDeltaR                       = 5
tauMatch.maxDPtRel                       = 99
tauMatch.resolveAmbiguities              = True
tauMatch.resolveByMatchQuality           = False
tauMatch.checkCharge                     = True
tauMatch.mcPdgId                         = [15]
tauMatch.mcStatus                        =  [2]

