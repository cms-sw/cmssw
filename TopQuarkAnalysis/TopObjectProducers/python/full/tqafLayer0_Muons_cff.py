import FWCore.ParameterSet.Config as cms

#
# L0 input
#

## import module
from PhysicsTools.PatAlgos.cleaningLayer0.muonCleaner_cfi import allLayer0Muons

## configure for tqaf
allLayer0Muons.muonSource               = 'muons'
allLayer0Muons.isolation.tracker.src    = 'patAODMuonIsolations:muIsoDepositTk'
allLayer0Muons.isolation.tracker.deltaR = 0.3
allLayer0Muons.isolation.tracker.cut    = 2.0
allLayer0Muons.isolation.hcal.src       = 'patAODMuonIsolations:muIsoDepositCalByAssociatorTowersecal'
allLayer0Muons.isolation.hcal.deltaR    = 0.3
allLayer0Muons.isolation.hcal.cut       = 2.0
allLayer0Muons.isolation.ecal.src       = 'patAODMuonIsolations:muIsoDepositCalByAssociatorTowershcal'
allLayer0Muons.isolation.ecal.deltaR    = 0.3
allLayer0Muons.isolation.ecal.cut       = 2.0

#
# genMatch
#

## import module
from PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi import muonMatch

## configure for tqaf
muonMatch.src                           = 'allLayer0Muons'
muonMatch.matched                       = 'genParticles'
muonMatch.maxDeltaR                     = 0.5
muonMatch.maxDPtRel                     = 0.5
muonMatch.resolveAmbiguities            = True
muonMatch.resolveByMatchQuality         = False
muonMatch.checkCharge                   = True
muonMatch.mcPdgId                       = [13]
muonMatch.mcStatus                      =  [1]
